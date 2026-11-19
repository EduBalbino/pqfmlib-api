"""Check offline SIH inference against an independent full statevector."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from qiskit.quantum_info import Statevector
from sklearn.decomposition import PCA


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('compact', [False, True])
@pytest.mark.parametrize('balanced,features,shape', [
    (False, 32, dict(gates=22, cones=8, width=6, depth=10)),
    (True, 32, dict(gates=36, cones=7, width=8, depth=12)),
    (False, 54, dict(gates=36, cones=7, width=8, depth=12)),
])
def test_offline_sih_features_and_partial_batch(tmp_path, monkeypatch, balanced, features, shape, compact):
    import tilelang
    from pqfmlib import HeisenbergProjectiveQFM
    from pqfmlib.experimental.heisenberg_sih_aot import build_sih_kernels, SIHHeisenbergAOT

    rng = np.random.default_rng(231)
    values = rng.normal(size=(9, features))
    values[8, :2] = (1000.0, -1000.0)
    model = HeisenbergProjectiveQFM(
        name_file='sih-aot-test', q_enc=19, R=2, alpha=.1, seed=42,
        encoding_mode='multi_axis', balanced_encoding=balanced,
        framework='tilelang', expectation_method='exact',
        save_circuit_drawings=False,
    ).fit(values)
    build_sih_kernels(tmp_path, batch=8, features=features, compact=compact, **shape)
    def no_jit(*args, **kwargs):
        raise AssertionError('runtime must use exported kernels')
    monkeypatch.setattr(tilelang, 'jit', no_jit)
    runner = SIHHeisenbergAOT(model, tmp_path)
    result = runner.transform(values)
    theta, _ = model._make_theta_matrix(values)
    prepared = model._prepared_execution
    for row in (0, 8):
        circuit = prepared.qc_t.assign_parameters(dict(zip(prepared.param_order, theta[row])))
        vector = Statevector.from_instruction(circuit).data
        reference = []
        for q in range(19):
            pair = vector.reshape(-1, 2, 1 << q)
            a, b = pair[:, 0, :], pair[:, 1, :]
            reference.extend([(np.abs(a)**2-np.abs(b)**2).sum(),
                              (2*(a.conj()*b).real).sum(),
                              (2*(a.conj()*b).imag).sum()])
        np.testing.assert_allclose(result[row], reference, rtol=0, atol=1e-9)
    np.testing.assert_allclose(result, runner.transform(values), rtol=0, atol=1e-12)
    gpu_result = runner.transform_device(torch.tensor(values, device='cuda'))
    assert gpu_result.is_cuda
    np.testing.assert_allclose(gpu_result.cpu().numpy(), result, rtol=0, atol=1e-12)
    import pqfmlib.experimental.heisenberg_sih_aot as aot
    monkeypatch.setattr(aot, 'SIHHeisenbergAOT', lambda fitted, path: runner)
    np.testing.assert_allclose(aot.transform_sih_aot(model, values), result, rtol=0, atol=1e-12)
    assert aot._RUNNERS[id(model)][1] is runner
    np.testing.assert_allclose(aot.transform_sih_aot(model, values), result, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match='finite PCA matrix'):
        runner.transform(np.full((1, features), np.nan))


def test_block_covariance_matches_dense_projections():
    path = (Path(__file__).resolve().parents[2] / 'experiments' / 'sih-sus'
            / 'sih_sus_reference_moe_sweep.ipynb')
    notebook = json.loads(path.read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if 'class OneHotCovariancePCA:' in ''.join(cell['source']))
    names = {'orient_pca_components', 'OneHotCovariancePCA'}
    definitions = [node for node in ast.parse(source).body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    scope = {'np': np}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), scope)

    rng = np.random.default_rng(42)
    numeric = rng.normal(size=(300, 4))
    first = np.eye(8)[rng.integers(0, 8, 300)]
    second = np.eye(5)[rng.integers(0, 5, 300)]
    X = np.column_stack([numeric, first, second])
    dense = PCA(n_components=12, svd_solver='covariance_eigh').fit(X[:250])
    block = scope['OneHotCovariancePCA'](12, 4).fit(X[:250])
    for model in (dense, block):
        scope['orient_pca_components'](model.components_)
    np.testing.assert_allclose(block.explained_variance_, dense.explained_variance_, atol=1e-12)
    np.testing.assert_allclose(block.transform(X[250:]), dense.transform(X[250:]), atol=1e-11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('qubits', [26, 33])
@pytest.mark.parametrize('mode,balanced,alpha', [
    ('multi_axis', False, .1),
    ('multi_axis', True, .1),
    ('shared_feature', False, .2),
])
def test_wide_offline_sih_against_bound_circuits(tmp_path, qubits, mode, balanced, alpha):
    from qiskit import QuantumCircuit

    from pqfmlib import HeisenbergProjectiveQFM
    from pqfmlib.experimental.heisenberg_sih_aot import (
        SIHHeisenbergAOT, build_sih_kernels, plan_sih_circuit,
    )

    values = np.random.default_rng(42).normal(size=(3, 50))
    model = HeisenbergProjectiveQFM(
        name_file='sih-aot-wide-test', q_enc=qubits, R=1, alpha=alpha, seed=42,
        encoding_mode=mode, balanced_encoding=balanced, framework='tilelang',
        expectation_method='exact', save_circuit_drawings=False,
    ).fit(values)
    shape, *_ = plan_sih_circuit(model)
    build_sih_kernels(tmp_path, batch=2, **shape)
    observed = SIHHeisenbergAOT(model, tmp_path).transform(values)
    theta, _ = model._make_theta_matrix(values)
    prepared = model._prepared_execution
    reference = []
    for angles in theta:
        circuit = prepared.qc_t.assign_parameters(dict(zip(prepared.param_order, angles)))
        gates = [(item.operation, tuple(circuit.find_bit(q).index for q in item.qubits))
                 for item in circuit.data if item.operation.name != 'barrier']
        row = []
        for target in range(qubits):
            # Use Qiskit's original bound gates, including identity rotations.
            active = {target}
            selected = []
            for op, sites in reversed(gates):
                if active.intersection(sites):
                    active.update(sites)
                    selected.append((op, sites))
            positions = {site: i for i, site in enumerate(sorted(active))}
            local = QuantumCircuit(len(positions))
            for op, sites in reversed(selected):
                local.append(op, [positions[site] for site in sites])
            vector = Statevector.from_instruction(local).data
            pair = vector.reshape(-1, 2, 1 << positions[target])
            a, b = pair[:, 0, :], pair[:, 1, :]
            row.extend([(np.abs(a)**2-np.abs(b)**2).sum(),
                        (2*(a.conj()*b).real).sum(),
                        (2*(a.conj()*b).imag).sum()])
        reference.append(row)
    np.testing.assert_allclose(observed, reference, rtol=0, atol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('topology,mode', [('full', 'commuting_x'), ('native', 'forest')])
def test_specialized_topology_readouts_match_bound_qiskit(tmp_path, topology, mode):
    import json

    from pqfmlib import HeisenbergProjectiveQFM
    from pqfmlib.experimental.sih_topology_sweep import (
        CommutingXSweepReadout, ForestSweepReadout, build_commuting_x, build_forest,
        commuting_x_plan, forest_plan,
    )

    snapshot = tmp_path / 'tree.json'
    snapshot.write_text(json.dumps({
        'schema_version': 1, 'coupling_edges': [[0, 1], [0, 3], [1, 2], [3, 4]],
    }))
    inputs = np.random.default_rng(514).normal(size=(3, 7 if topology == 'full' else 10))
    native = dict(ideal=False, topology_file=str(snapshot)) if topology == 'native' else {}
    model = HeisenbergProjectiveQFM(
        name_file='sih-topology-test', q_enc=5, topology=topology, R=1, alpha=.2,
        framework='tilelang', expectation_method='exact', encoding_mode='multi_axis',
        balanced_encoding=True, **native,
    ).fit(inputs)
    if mode == 'commuting_x':
        degree = commuting_x_plan(model)[1].shape[1]
        build_commuting_x(tmp_path, 5, inputs.shape[1], degree, batch=2, threads=32)
        reader = CommutingXSweepReadout(model, tmp_path)
    else:
        edges = len(forest_plan(model)[1])
        build_forest(tmp_path, 5, inputs.shape[1], edges, batch=2, threads=32)
        reader = ForestSweepReadout(model, tmp_path)
    angles, _ = model._make_theta_matrix(inputs)
    prepared = model._prepared_execution
    reference = []
    for row in angles:
        circuit = prepared.qc_t.assign_parameters(dict(zip(prepared.param_order, row)))
        state = Statevector.from_instruction(circuit)
        reference.append([state.expectation_value(group[0]).real for group in prepared.obs_isa_broadcast])
    np.testing.assert_allclose(reader.transform(inputs), reference, rtol=0, atol=1e-11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('wide', [False, True])
def test_frontier_matches_bound_qiskit_with_reused_slots(tmp_path, wide):
    from pqfmlib import HeisenbergProjectiveQFM
    from pqfmlib.experimental.sih_frontier_tilelang import (
        FrontierSweepReadout, build_frontier, frontier_plan, frontier_shape,
    )

    inputs = np.random.default_rng(352).normal(size=(5, 11))
    model = HeisenbergProjectiveQFM(
        name_file='sih-frontier-test', q_enc=7, topology='chain' if wide else 'full', R=1, alpha=.2,
        framework='tilelang', expectation_method='exact', encoding_mode='shared_feature',
    ).fit(inputs)
    plan = frontier_plan(model, max_width=7 if wide else 5, compact=wide)
    assert plan is not None and plan[0] < model.q_enc
    if wide:
        from pqfmlib.experimental.sih_wide_frontier_tilelang import build_wide_frontier, wide_frontier_shape

        build_wide_frontier(tmp_path, *wide_frontier_shape(model, plan), batch=3, threads=64, compact=True)
    else:
        build_frontier(tmp_path, *frontier_shape(model, plan), batch=3, threads=64)
    angles, _ = model._make_theta_matrix(inputs)
    prepared = model._prepared_execution
    expected = []
    for row in angles:
        circuit = prepared.qc_t.assign_parameters(dict(zip(prepared.param_order, row)))
        state = Statevector.from_instruction(circuit)
        expected.append([state.expectation_value(group[0]).real for group in prepared.obs_isa_broadcast])
    np.testing.assert_allclose(FrontierSweepReadout(model, tmp_path).transform(inputs),
                               expected, rtol=0, atol=1e-11)
