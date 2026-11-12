import numpy as np
import pytest


@pytest.mark.parametrize('size,batch', [(256, 1), (1024, 3)])
def test_cone_expectations_match_numpy(size, batch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('tilelang')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    from pqfmlib.experimental.heisenberg_tilelang import cone_expectations

    rng = np.random.default_rng(42)
    cones, ids, targets = 2, [0, 1, 0], [0, int(np.log2(size)) - 1, 2]
    state = rng.normal(size=(cones * batch, size, 2)) / np.sqrt(size)
    expected = np.empty((batch, len(ids), 3))
    for obs, (cone, target) in enumerate(zip(ids, targets)):
        idx = np.arange(size // 2)
        zero = (idx & ((1 << target) - 1)) | ((idx >> target) << (target + 1))
        a = state[cone*batch:(cone+1)*batch, zero, :]
        b = state[cone*batch:(cone+1)*batch, zero | (1 << target), :]
        ar, ai, br, bi = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
        expected[:, obs, 0] = (ar*ar + ai*ai - br*br - bi*bi).sum(axis=1)
        expected[:, obs, 1] = (2*(ar*br + ai*bi)).sum(axis=1)
        expected[:, obs, 2] = (2*(ar*bi - ai*br)).sum(axis=1)
    actual = cone_expectations(torch.tensor(state, device='cuda'), batch, cones, ids, targets)
    np.testing.assert_allclose(actual.cpu().numpy(), expected.reshape(batch, -1), rtol=0, atol=1e-9)


def test_tilelang_rejects_unplanned_parameter_binding(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('tilelang')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    from pqfmlib import HeisenbergProjectiveQFM
    from pqfmlib.experimental import heisenberg_tilelang as backend
    x = np.zeros((1, 6))
    model = HeisenbergProjectiveQFM(name_file='strict_test', q_enc=3, R=2, encoding_mode='multi_axis', framework='tilelang', expectation_method='exact', save_circuit_drawings=False).fit(x)
    monkeypatch.setattr(backend, 'affine_rotation_plan', lambda *args: None)
    with pytest.raises(ValueError, match='refusing slow per-row parameter binding'):
        model.transform(x)


def test_tilelang_rejects_unrequested_pair_full_state(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('tilelang')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is required')
    from pqfmlib import HeisenbergProjectiveQFM
    x = np.zeros((1, 6))
    model = HeisenbergProjectiveQFM(name_file='strict_pair_test', q_enc=3, R=2, encoding_mode='multi_axis', measure_2local_diagonal=True, framework='tilelang', expectation_method='exact', save_circuit_drawings=False).fit(x)
    with pytest.raises(RuntimeError, match='refusing full-state fallback'):
        model.transform(x)


@pytest.mark.parametrize('qubits', [4, 6])
def test_identity_gate_pruning_preserves_original_circuit(qubits):
    pytest.importorskip('tilelang')
    from pqfmlib import HeisenbergProjectiveQFM
    from pqfmlib.experimental.heisenberg_tilelang import affine_rotation_plan, circuit_gates
    from qiskit.quantum_info import Operator
    x = np.random.default_rng(8).normal(size=(2, 3 * qubits))
    model = HeisenbergProjectiveQFM(name_file='pruning', q_enc=qubits, R=2,
        alpha=.1, seed=42, encoding_mode='multi_axis', expectation_method='statevector',
        save_circuit_drawings=False).fit(x)
    prepared = model._prepared_execution
    theta, _ = model._make_theta_matrix(x)
    plan = affine_rotation_plan(prepared.qc_t, prepared.param_order)
    _, full = circuit_gates(prepared.qc_t, prepared.param_order, theta, plan, prepared.qc_t.data, strip_identities=False)
    _, pruned = circuit_gates(prepared.qc_t, prepared.param_order, theta, plan, prepared.qc_t.data)
    assert len(pruned) < len(full)
    # Every discarded operation is exactly the identity for every row.
    identities = [gate for gate in full if len(gate[0]) == 2 and np.array_equal(gate[2], np.broadcast_to(np.eye(4), gate[2].shape))]
    assert len(full) - len(pruned) == len(identities)
    assert all(np.allclose(matrix[0], Operator(np.eye(4)).data, rtol=0, atol=0) for _, _, matrix in identities)
