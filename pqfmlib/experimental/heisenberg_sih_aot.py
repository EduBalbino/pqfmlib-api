"""Offline-compiled PCA-32 and PCA-54 SIH-SUS Heisenberg inference.

The builder runs on the target GPU; inference loads its exported libraries and
never invokes the TileLang compiler. A runner owns reusable CUDA workspaces and
is not thread-safe. This deliberately rejects other circuit shapes. Build with
``build_sih_kernels(path, batch=320)``, then construct
``SIHHeisenbergAOT(fitted_map, path)`` and pass PCA rows to ``transform``.
``transform_device`` accepts and returns CUDA tensors to avoid host transfers.
Run this module as a script to build both notebook artifacts on the target GPU.
"""
import json
from pathlib import Path
from weakref import ref

import numpy as np
import torch

_RUNNERS = {}


def build_sih_kernels(directory, *, batch=320, gates=22, cones=8, width=6, depth=10, features=32, qubits=19, compact=False, threads=128):
    """Compile one fixed SIH cone shape offline."""
    import tilelang
    import tilelang.language as T

    if min(batch, gates, cones, width, depth, features, qubits) <= 0:
        raise ValueError("kernel dimensions must be positive")
    if compact and (threads < 32 or threads % 32):
        raise ValueError("packed cones require full CUDA warps")
    size, observables = 1 << width, qubits
    output_stride = 3 * (1 + threads // 32) if compact else 3

    @tilelang.jit(target="cuda", execution_backend="tvm_ffi")
    def coefficients():
        @T.prim_func
        def kernel(values: T.Tensor((batch, features), "float64"),
                   columns: T.Tensor((gates, 3), "int32"),
                   affine: T.Tensor((gates, 3, 2), "float64"),
                   coeff: T.Tensor((gates, batch, 2, 2, 2), "float64")):
            with T.Kernel(T.ceildiv(gates * batch, 128), threads=128) as block:
                for lane in T.Parallel(128):
                    i = block * 128 + lane
                    if i < gates * batch:
                        g, sample = i // batch, i % batch
                        angles = T.alloc_local((3,), "float64")
                        for axis in T.serial(3):
                            col = columns[g, axis]
                            v = T.if_then_else(col >= 0, values[sample, T.max(col, 0)], 0.0)
                            v = T.tanh(v / 3.0) * 6.283185307179586
                            angles[axis] = (v * affine[g, axis, 0] + affine[g, axis, 1]) * 0.5
                        z, x, y = angles[0], angles[1], angles[2]
                        for parity in T.serial(2):
                            phase_im = T.if_then_else(parity == 0, -T.sin(z), T.sin(z))
                            phase_re = T.cos(z)
                            angle = T.if_then_else(parity == 0, x-y, x+y)
                            c, s = T.cos(angle), T.sin(angle)
                            coeff[g, sample, parity, 0, 0] = phase_re*c
                            coeff[g, sample, parity, 0, 1] = phase_im*c
                            coeff[g, sample, parity, 1, 0] = phase_im*s
                            coeff[g, sample, parity, 1, 1] = -phase_re*s
        return kernel

    # Each thread owns both members of a parity pair. The kernel supplies
    # block barriers between successive gates and at the warp reduction.
    @tilelang.jit(target="cuda", execution_backend="tvm_ffi",
                  pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: compact})
    def evolve_and_measure():
        @T.prim_func
        def kernel(initial: T.Tensor((cones, size, 2), "float64"),
                   coeff: T.Tensor((gates, batch, 2, 2, 2), "float64"),
                   sites: T.Tensor((depth + int(compact), cones, 3), "int32"),
                   obs_cones: T.Tensor((observables,), "int32"),
                   targets: T.Tensor((observables,), "int32"),
                   output: T.Tensor((batch, observables, output_stride), "float64")):
            with T.Kernel(cones * batch, threads=threads) as b:
                state = T.alloc_shared((size, 2), "float64")
                if not compact:
                    nxt = T.alloc_shared((size, 2), "float64")
                if compact:
                    totals = T.alloc_local((3,), "float64")
                    tid = T.get_thread_binding()
                else:
                    z = T.alloc_fragment((1, size // 2), "float64")
                    x = T.alloc_fragment((1, size // 2), "float64")
                    y = T.alloc_fragment((1, size // 2), "float64")
                    rz = T.alloc_fragment((1,), "float64")
                    rx = T.alloc_fragment((1,), "float64")
                    ry = T.alloc_fragment((1,), "float64")
                cone, sample = b // batch, b % batch
                active = size
                if compact:
                    active = 1 << sites[depth, cone, 0]
                    for chunk in T.serial(T.ceildiv(active, threads)):
                        i = chunk * threads + tid
                        if i < active:
                            state[i, 0], state[i, 1] = initial[cone, i, 0], initial[cone, i, 1]
                    T.sync_threads()
                    for step in T.serial(depth):
                        q0, q1, gate = sites[step, cone, 0], sites[step, cone, 1], sites[step, cone, 2]
                        if gate >= 0:
                            for chunk in T.serial(T.ceildiv(active // 2, threads)):
                                lane = chunk * threads + tid
                                if lane < active // 2:
                                    low = (lane & ((1 << q0) - 1)) | ((lane >> q0) << (q0 + 1))
                                    high = low ^ (1 << q0) ^ (1 << q1)
                                    p = (low >> q1) & 1
                                    ar, ai = state[low, 0], state[low, 1]
                                    br, bi = state[high, 0], state[high, 1]
                                    dr, di = coeff[gate, sample, p, 0, 0], coeff[gate, sample, p, 0, 1]
                                    er, ei = coeff[gate, sample, p, 1, 0], coeff[gate, sample, p, 1, 1]
                                    state[low, 0] = dr*ar - di*ai + er*br - ei*bi
                                    state[low, 1] = dr*ai + di*ar + er*bi + ei*br
                                    state[high, 0] = dr*br - di*bi + er*ar - ei*ai
                                    state[high, 1] = dr*bi + di*br + er*ai + ei*ar
                            T.sync_threads()
                else:
                    for i in T.Parallel(size):
                        state[i, 0], state[i, 1] = initial[cone, i, 0], initial[cone, i, 1]
                    T.sync_threads()
                    for step in T.serial(depth):
                        for i in T.Parallel(size):
                            q0, q1, gate = sites[step, cone, 0], sites[step, cone, 1], sites[step, cone, 2]
                            if gate < 0:
                                nxt[i, 0], nxt[i, 1] = state[i, 0], state[i, 1]
                            else:
                                parity = ((i >> q0) ^ (i >> q1)) & 1
                                source = i ^ (1 << q0) ^ (1 << q1)
                                ar, ai = state[i, 0], state[i, 1]
                                br, bi = state[source, 0], state[source, 1]
                                dr, di = coeff[gate, sample, parity, 0, 0], coeff[gate, sample, parity, 0, 1]
                                er, ei = coeff[gate, sample, parity, 1, 0], coeff[gate, sample, parity, 1, 1]
                                nxt[i, 0] = dr*ar - di*ai + er*br - ei*bi
                                nxt[i, 1] = dr*ai + di*ar + er*bi + ei*br
                        T.sync_threads()
                        for i in T.Parallel(size):
                            state[i, 0], state[i, 1] = nxt[i, 0], nxt[i, 1]
                        T.sync_threads()
                for obs in T.serial(observables):
                    if obs_cones[obs] == cone:
                        if compact:
                            for axis in T.unroll(3):
                                totals[axis] = 0.0
                            target = targets[obs]
                            for chunk in T.serial(T.ceildiv(active // 2, threads)):
                                lane = chunk * threads + tid
                                if lane < active // 2:
                                    zero = (lane & ((1 << target)-1)) | ((lane >> target) << (target+1))
                                    one = zero | (1 << target)
                                    ar, ai = state[zero, 0], state[zero, 1]
                                    br, bi = state[one, 0], state[one, 1]
                                    totals[0] += ar*ar + ai*ai - br*br - bi*bi
                                    totals[1] += 2*(ar*br + ai*bi)
                                    totals[2] += 2*(ar*bi - ai*br)
                            for axis in T.unroll(3):
                                value = T.warp_reduce_sum(totals[axis])
                                if tid % 32 == 0:
                                    output[sample, obs, 3 + (tid // 32) * 3 + axis] = value
                            T.sync_threads()
                            if tid < 3:
                                total = T.alloc_local((1,), "float64")
                                total[0] = 0.0
                                for warp in T.unroll(threads // 32):
                                    total[0] += output[sample, obs, 3 + warp * 3 + tid]
                                output[sample, obs, tid] = total[0]
                        else:
                            for lane in T.Parallel(size // 2):
                                z[0, lane], x[0, lane], y[0, lane] = 0.0, 0.0, 0.0
                                if lane < active // 2:
                                    target = targets[obs]
                                    zero = (lane & ((1 << target)-1)) | ((lane >> target) << (target+1))
                                    one = zero | (1 << target)
                                    ar, ai = state[zero, 0], state[zero, 1]
                                    br, bi = state[one, 0], state[one, 1]
                                    z[0, lane] = ar*ar + ai*ai - br*br - bi*bi
                                    x[0, lane] = 2*(ar*br + ai*bi)
                                    y[0, lane] = 2*(ar*bi - ai*br)
                            T.reduce_sum(z, rz, dim=1)
                            T.reduce_sum(x, rx, dim=1)
                            T.reduce_sum(y, ry, dim=1)
                            output[sample, obs, 0] = rz[0]
                            output[sample, obs, 1] = rx[0]
                            output[sample, obs, 2] = ry[0]
        return kernel

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    coefficients().export_library(str(path / 'coefficients.so'))
    evolve_and_measure().export_library(str(path / 'evolve_and_measure.so'))
    (path / 'metadata.json').write_text(json.dumps({'batch': batch, 'qubits': qubits, 'features': features, 'gates': gates, 'cones': cones, 'width': width, 'depth': depth, 'cuda_capability': list(torch.cuda.get_device_capability()), 'tilelang': tilelang.__version__, 'compact': compact, 'threads': threads}, indent=2))


def plan_sih_circuit(fmap):
    """Derive the exact active-gate and causal-cone layout from the fitted map."""
    from pqfmlib.experimental.heisenberg_tilelang import (
        affine_rotation_plan, circuit_gates, causal_cones,
        is_heisenberg_group, product_cone_initial_states,
    )
    from pqfmlib.maps.heisenberg import heisenberg_chain_observables

    qubits, features = fmap.q_enc, fmap.num_features
    if (not fmap.use_tanh_scaling or fmap.coupling_mode != 'direct'
            or fmap.measure_2local_diagonal or fmap.measure_cross_observables):
        raise ValueError('AOT requires direct, tanh-scaled local-observable circuits')
    prepared = fmap._prepared_execution
    circuit, params = prepared.qc_t, prepared.param_order
    expected_obs, expected_metadata = heisenberg_chain_observables(qubits)
    actual_obs = prepared.obs_isa_broadcast
    if (prepared.obs_metadata != expected_metadata or len(actual_obs) != len(expected_obs)
            or any(len(group) != 1 or not group[0].equiv(obs)
                   for group, obs in zip(actual_obs, expected_obs))):
        raise ValueError('AOT artifact requires local Z/X/Y observables in site order')
    plan = affine_rotation_plan(circuit, params)
    if plan is None:
        raise ValueError('circuit has non-affine rotation angles')
    probe, _ = fmap._make_theta_matrix(np.linspace(.1, 1.0, features)[None, :])
    slots = np.flatnonzero(probe[0])
    if len(slots) != features or probe.shape[1] != len(params):
        raise ValueError('expected exactly one parameter slot per PCA input')
    slot_columns = np.full(len(params), -1, dtype=np.int32)
    slot_columns[slots[np.argsort(probe[0, slots])]] = np.arange(features)
    instructions = tuple(circuit.data)
    _, all_gates = circuit_gates(circuit, params, probe, plan, instructions)
    cones, obs_cones, targets, width = causal_cones(all_gates, qubits)
    columns, affine, active_sites = [], [], []
    i = 0
    while i < len(instructions):
        item = instructions[i]
        if item.operation.name in ('barrier', 'unitary'):
            i += 1
        elif is_heisenberg_group(instructions[i:i+3]):
            group = [plan[j] for j in range(i, i+3)]
            if any((col is not None and slot_columns[col] >= 0 and scale != 0) or offset != 0
                   for col, scale, offset in group):
                columns.append([int(slot_columns[col]) if col is not None else -1
                                for col, _, _ in group])
                affine.append([[scale, offset] for _, scale, offset in group])
                active_sites.append([circuit.find_bit(q).index for q in item.qubits])
            i += 3
        else:
            raise ValueError('unsupported gate in AOT circuit')
    count, cone_count = len(columns), len(cones)
    if (len(all_gates) != qubits + count
            or any(sites != active_sites[k] for k, (sites, _, _) in enumerate(all_gates[qubits:]))
            or any(len(sites) != 1 for sites, _, _ in all_gates[:qubits])
            or {sites[0] for sites, _, _ in all_gates[:qubits]} != set(range(qubits))):
        raise ValueError('circuit does not match SIH cone plan')
    initial_by_site = {sites[0]: matrix[0, :, 0] for sites, _, matrix in all_gates[:qubits]}
    local = np.zeros((cone_count, width, 2), complex)
    local[:, :, 0] = 1
    depth = max(sum(idx >= qubits for idx in selected) for _, selected, _ in cones)
    sites = np.full((depth, cone_count, 3), -1, np.int32)
    for k, (positions, selected, _) in enumerate(cones):
        for site, pos in positions.items():
            local[k, pos] = initial_by_site[site]
        for step, idx in enumerate(idx for idx in selected if idx >= qubits):
            a, b = all_gates[idx][0]
            sites[step, k] = positions[a], positions[b], idx-qubits
    initial = product_cone_initial_states(local.shape, local.tobytes())
    shape = dict(features=features, qubits=qubits, gates=count, cones=cone_count,
                 width=width, depth=depth)
    return shape, initial, sites, columns, affine, obs_cones, targets


class SIHHeisenbergAOT:
    """Frozen plan and CUDA workspaces for a supported SIH map."""

    def __init__(self, fmap, directory):
        path = Path(directory)
        config = json.loads((path / 'metadata.json').read_text())
        if list(torch.cuda.get_device_capability()) != config['cuda_capability']:
            raise RuntimeError('AOT artifact CUDA capability mismatch')
        shape, initial, sites, columns, affine, obs_cones, targets = plan_sih_circuit(fmap)
        config.setdefault('qubits', 19)
        config.setdefault('features', 32)
        if any(config[key] != size for key, size in shape.items() if key != 'depth'):
            raise ValueError(f'circuit shape {shape} does not match compiled SIH artifact')
        if shape['depth'] > config['depth']:
            raise ValueError('cone exceeds compiled depth')
        if shape['depth'] < config['depth']:
            sites = np.pad(sites, ((0, config['depth']-shape['depth']), (0, 0), (0, 0)),
                           constant_values=-1)
        if config.get('compact'):
            # Each cone has a dense local site numbering. Isolated observables
            # contribute a site even when the cone contains no pair gates.
            widths = np.maximum(sites[:, :, :2].max(axis=(0, 2)) + 1, 1)
            np.maximum.at(widths, obs_cones, np.asarray(targets) + 1)
            sites = np.concatenate((sites, np.broadcast_to(widths[:, None], (1, len(widths), 3))))
        self.features = features = shape['features']
        self.qubits = shape['qubits']
        count = shape['gates']
        self.batch = config['batch']
        stride = 3 * (1 + config['threads'] // 32) if config.get('compact') else 3
        from pqfmlib.experimental.sih_topology_sweep import _load_offline_kernel

        self.coefficients = _load_offline_kernel(path / 'coefficients.so')
        self.evolve = _load_offline_kernel(path / 'evolve_and_measure.so')
        self.initial = torch.view_as_real(torch.tensor(initial, device='cuda')).contiguous()
        self.sites = torch.tensor(sites, device='cuda')
        self.columns = torch.tensor(columns, dtype=torch.int32, device='cuda')
        self.affine = torch.tensor(affine, dtype=torch.float64, device='cuda')
        self.obs_cones = torch.tensor(obs_cones, dtype=torch.int32, device='cuda')
        self.targets = torch.tensor(targets, dtype=torch.int32, device='cuda')
        self.input = torch.empty((self.batch, features), dtype=torch.float64, device='cuda')
        self.coeff = torch.empty((count, self.batch, 2, 2, 2), dtype=torch.float64, device='cuda')
        self.output = torch.empty((self.batch, self.qubits, stride), dtype=torch.float64, device='cuda')

    def transform_device(self, values):
        """Return all local features on CUDA; input must be a CUDA FP64 matrix."""
        if values.device != self.input.device or values.dtype != torch.float64 or values.ndim != 2 or values.shape[1] != self.features:
            raise ValueError('expected CUDA float64 PCA matrix with the compiled input width')
        result = torch.empty((len(values), 3*self.qubits), dtype=torch.float64, device=values.device)
        for start in range(0, len(values), self.batch):
            count = min(self.batch, len(values)-start)
            self.input[:count].copy_(values[start:start+count])
            if count < self.batch:
                self.input[count:].zero_()
            self.coefficients(self.input, self.columns, self.affine, self.coeff)
            self.evolve(self.initial, self.coeff, self.sites, self.obs_cones, self.targets, self.output)
            result[start:start+count].copy_(self.output[:count, :, :3].reshape(count, 3*self.qubits))
        return result

    def transform(self, values):
        """Transfer PCA rows once, and return feature rows as a NumPy array."""
        array = np.asarray(values, dtype=np.float64, order='C')
        if array.ndim != 2 or array.shape[1] != self.features or not np.isfinite(array).all():
            raise ValueError('expected a finite PCA matrix with the compiled input width')
        return self.transform_device(torch.as_tensor(array, device='cuda')).cpu().numpy()


def transform_sih_aot(fmap, pca_values):
    """Reuse a compiled runner per fitted map without serializing CUDA state."""
    key = id(fmap)
    cached = _RUNNERS.get(key)
    if cached is None or cached[0]() is not fmap:
        if fmap.num_features == 54 and not fmap.balanced_encoding:
            artifact = 'q19-pca54-b320'
        else:
            artifact = ('q19-pca32-balanced-b320' if fmap.balanced_encoding
                        else 'q19-pca32-b320-d10')
        root = Path(__file__).resolve().parents[2] / 'experiments' / 'sih-sus' / 'aot'
        runner = SIHHeisenbergAOT(fmap, root / artifact)
        _RUNNERS[key] = (ref(fmap, lambda _: _RUNNERS.pop(key, None)), runner)
    else:
        runner = cached[1]
    return runner.transform(pca_values)


if __name__ == '__main__':
    output = Path(__file__).resolve().parents[2] / 'experiments' / 'sih-sus' / 'aot'
    build_sih_kernels(output / 'q19-pca32-b320-d10')
    build_sih_kernels(output / 'q19-pca32-balanced-b320', gates=36, cones=7, width=8, depth=12)
    build_sih_kernels(output / 'q19-pca54-b320', features=54, gates=36, cones=7, width=8, depth=12)
