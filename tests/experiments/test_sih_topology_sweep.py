import json

import numpy as np
import pytest

from pqfmlib.experimental.sih_topology_sweep import locked_sweep_journal, topology_profiles


def test_complete_sweep_covers_every_strength_for_each_encoding(tmp_path):
    profiles = topology_profiles(tmp_path, complete=True)

    assert len(profiles) == 90
    for qubits in (19, 26):
        for topology, layout in (
            ('chain', 'auto'), ('plaquettes', 'auto'), ('full', 'auto'),
            ('native', 'auto'), ('native', 'fixed'),
        ):
            for mode in ('axes', 'balanced', 'shared'):
                prefix = f'q{qubits}_{topology}_{layout}_{mode}_s'
                strengths = {strength for name, (_, strength) in profiles.items() if name.startswith(prefix)}
                assert strengths == {.05, .10, .20}


def test_resume_recovers_a_torn_tail_and_excludes_a_second_writer(tmp_path):
    journal = tmp_path / 'results.jsonl'
    complete = (json.dumps({'name': 'complete'}) + '\n').encode()
    journal.write_bytes(complete + b'{"name": "interrupted')

    with locked_sweep_journal(journal) as records:
        assert records == {'complete': {'name': 'complete'}}
        assert journal.read_bytes() == complete
        with pytest.raises(BlockingIOError), locked_sweep_journal(journal):
            pytest.fail('a second writer acquired the sweep journal')

    with locked_sweep_journal(journal) as records:
        assert list(records) == ['complete']


def test_dense_readout_loads_offline_kernels_without_compiling(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from pqfmlib import HeisenbergProjectiveQFM
    from pqfmlib.experimental.sih_topology_sweep import GenericTileLangSweepReadout

    rows = np.random.default_rng(42).normal(size=(3, 6))
    model = HeisenbergProjectiveQFM(
        name_file='dense-offline-test', q_enc=4, topology='full', encoding_mode='shared_feature',
        framework='tilelang', expectation_method='exact', save_circuit_drawings=False,
    ).fit(rows)
    builder = GenericTileLangSweepReadout(model)
    executor = builder.executor
    executor.matrix_backend = 'cpu'
    launch = executor._launcher

    def export(kernel, *signature):
        result = launch(kernel, *signature)
        path = tmp_path / ('_'.join(map(str, (kernel, *signature))) + '.so')
        if not path.exists():
            result.export_library(str(path))
        return result

    executor._launcher = export
    expected = builder.transform(rows)
    offline = GenericTileLangSweepReadout(model, artifacts=tmp_path)
    monkeypatch.setattr(offline.executor, '_kernels', None)
    np.testing.assert_allclose(offline.transform(rows), expected, rtol=0, atol=1e-12)
