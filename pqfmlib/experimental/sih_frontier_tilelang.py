"""Exact density-matrix frontier for full SIH Heisenberg circuits."""

import json
from functools import lru_cache
from itertools import groupby
from pathlib import Path

import numpy as np


def _edge_dependencies(edges, qubits):
    wires = [0] * qubits
    predecessors = []
    previous = [0] * qubits
    for i, (left, right) in enumerate(edges):
        bit = 1 << i
        predecessors.append(previous[left] | previous[right])
        previous[left] = previous[right] = bit
        wires[left] |= bit
        wires[right] |= bit
    return tuple(wires), tuple(predecessors)


@lru_cache(maxsize=128)
def _bounded_order(edges, qubits, width=5):
    """Search topological schedules subject to a live-wire bound."""
    wires, predecessors = _edge_dependencies(edges, qubits)
    all_gates = (1 << len(edges)) - 1
    dead_ends = set()

    def visit(done):
        if done == all_gates:
            return ()
        if done in dead_ends or len(dead_ends) >= 20_000:
            return None
        live = {q for q, gates in enumerate(wires) if gates & done and gates & ~done}
        ready = []
        for i, (a, b) in enumerate(edges):
            bit = 1 << i
            if done & bit or predecessors[i] & ~done:
                continue
            active = live | {a, b}
            if len(active) <= width:
                next_done = done | bit
                after = sum(bool(wires[q] & ~next_done) for q in active)
                ready.append((len(active), after, i, next_done))
        for _, _, i, next_done in sorted(ready):
            remainder = visit(next_done)
            if remainder is not None:
                return (i, *remainder)
        dead_ends.add(done)
        return None

    return visit(0)


@lru_cache(maxsize=128)
def _compact_order(edges, qubits, width):
    """Find a bounded schedule that minimizes the number of touched parity blocks."""
    wires, parents = _edge_dependencies(edges, qubits)
    wave = {0: (0, ())}
    for _ in edges:
        candidates = {}
        for done, (cost, order) in wave.items():
            live = {q for q, gates in enumerate(wires) if gates & done and gates & ~done}
            for index, edge in enumerate(edges):
                bit = 1 << index
                if done & bit or parents[index] & ~done:
                    continue
                active = live.union(edge)
                if len(active) > width:
                    continue
                next_done = done | bit
                before, peak = len(live), len(active)
                after = sum(bool(wires[q] & ~next_done) for q in active)
                work = sum(4**rank for rank in range(before, peak))  # Attach
                work += 4**(peak-1)  # Interact
                work += sum(4**rank for rank in range(after, peak))  # Trace out
                last = candidates.get(next_done)
                if last is None or cost + work < last[0]:
                    candidates[next_done] = (cost + work, (*order, index))
        if not candidates:
            return _bounded_order(edges, qubits, width)
        wave = dict(sorted(candidates.items(), key=lambda item: (item[1][0], item[0]))[:128])
    return next(iter(wave.values()))[1]


def frontier_plan(model, *, max_width=5, compact=False):
    """Reorder disjoint-wire gates and retire wires after their final interaction."""
    from qiskit.quantum_info import Operator
    from pqfmlib.experimental.sih_topology_sweep import active_circuit, _input_columns

    circuit = active_circuit(model)
    qubits = model.q_enc
    columns = _input_columns(model)

    def sites(instruction):
        return tuple(circuit.find_bit(q).index for q in instruction.qubits)

    initial = np.empty((qubits, 2, 2, 2), dtype=np.float64)
    for instruction in circuit.data[:qubits]:
        (site,) = sites(instruction)
        vector = Operator(instruction.operation).data[:, 0]
        rho = np.outer(vector, vector.conj())
        initial[site] = np.stack([rho.real, rho.imag], axis=-1)

    edges, inputs, scales = [], [], []
    for edge, gates in groupby(circuit.data[qubits:], key=sites):
        indices = np.zeros(3, dtype=np.int32)
        weights = np.zeros(3)
        for instruction in gates:
            op = instruction.operation
            if op.name not in ('rxx', 'ryy', 'rzz'):
                return None
            axis = 'xyz'.index(op.name[-1])
            angle = op.params[0]
            if len(angle.parameters) != 1 or weights[axis] != 0:
                return None
            parameter, = angle.parameters
            if float(angle.bind({parameter: 0})) != 0:
                return None
            indices[axis], weights[axis] = columns[parameter], float(angle.gradient(parameter))
        edges.append(edge)
        inputs.append(indices)
        scales.append(weights)

    degree = np.bincount(np.asarray(edges).ravel(), minlength=qubits)
    hubs = {q for q in range(qubits) if degree[q] >= qubits // 2}

    def order(index):
        leaves = tuple(q for q in edges[index] if q not in hubs)
        return (len(leaves), leaves, index)

    ordered = sorted(range(len(edges)), key=order) if model.topology == 'full' else None
    if ordered is None or any(
        [i for i in ordered if q in edges[i]] != [i for i, edge in enumerate(edges) if q in edge]
        for q in range(qubits)
    ):
        planner = _compact_order if compact else _bounded_order
        ordered = planner(tuple(edges), qubits, max_width)
        if ordered is None:
            return None

    actions, live = [], {}
    remaining = degree.copy()
    width = 0
    for index in ordered:
        edge = edges[index]
        for q in edge:
            if q not in live:
                slot = min(set(range(qubits)) - set(live.values()))
                live[q] = slot
                width = max(width, slot + 1)
                actions.append((0, slot, q, 0))
        actions.append((1, live[edge[0]], live[edge[1]], index))
        for q in edge:
            remaining[q] -= 1
            if remaining[q] == 0:
                actions.append((2, live.pop(q), q, 0))
    for q in np.flatnonzero(degree == 0):
        actions.extend(((0, 0, q, 0), (2, 0, q, 0)))
        width = max(width, 1)
    if not 1 <= width <= max_width:
        return None
    return width, tuple(np.ascontiguousarray(item) for item in
                        (initial, np.asarray(inputs, np.int32), np.asarray(scales, np.float64),
                         np.asarray(actions, np.int32)))


def build_frontier(directory, qubits, features, width, groups, steps, *, batch=256, threads=128):
    import tilelang
    import tilelang.language as T

    dim = 1 << width

    @tilelang.jit(target='cuda', execution_backend='tvm_ffi')
    def frontier():
        @T.prim_func
        def kernel(values: T.Tensor((batch, features), 'float64'),
                   initial: T.Tensor((qubits, 2, 2, 2), 'float64'),
                   columns: T.Tensor((groups, 3), 'int32'),
                   scales: T.Tensor((groups, 3), 'float64'),
                   actions: T.Tensor((steps, 4), 'int32'),
                   result: T.Tensor((batch, qubits, 3), 'float64')):
            with T.Kernel(batch, threads=threads) as row:
                rho = T.alloc_shared((dim, dim, 2), 'float64')
                work = T.alloc_shared((dim, dim, 2), 'float64')
                coeff = T.alloc_shared((2, 2, 2), 'float64')
                z = T.alloc_fragment((1, dim // 2), 'float64')
                x = T.alloc_fragment((1, dim // 2), 'float64')
                y = T.alloc_fragment((1, dim // 2), 'float64')
                rz = T.alloc_fragment((1,), 'float64')
                rx = T.alloc_fragment((1,), 'float64')
                ry = T.alloc_fragment((1,), 'float64')
                for i, j in T.Parallel(dim, dim):
                    rho[i, j, 0] = T.if_then_else(i == 0 and j == 0, 1.0, 0.0)
                    rho[i, j, 1] = 0.0
                T.sync_threads()
                for step in T.serial(steps):
                    kind, a, b, edge = actions[step, 0], actions[step, 1], actions[step, 2], actions[step, 3]
                    bit = 1 << a
                    if kind == 1:
                        flip = bit | (1 << b)
                        for parity in T.Parallel(2):
                            ax = T.float64(3.141592653589793) * T.tanh(values[row, columns[edge, 0]] / 3.0) * scales[edge, 0]
                            ay = T.float64(3.141592653589793) * T.tanh(values[row, columns[edge, 1]] / 3.0) * scales[edge, 1]
                            az = T.float64(3.141592653589793) * T.tanh(values[row, columns[edge, 2]] / 3.0) * scales[edge, 2]
                            angle = T.if_then_else(parity == 0, ax - ay, ax + ay)
                            cr, ci = T.cos(az), T.if_then_else(parity == 0, -T.sin(az), T.sin(az))
                            ca, sa = T.cos(angle), T.sin(angle)
                            coeff[parity, 0, 0], coeff[parity, 0, 1] = cr * ca, ci * ca
                            coeff[parity, 1, 0], coeff[parity, 1, 1] = ci * sa, -cr * sa
                        T.sync_threads()
                        for i, j in T.Parallel(dim, dim):
                            p = ((i >> a) ^ (i >> b)) & 1
                            ar, ai = rho[i, j, 0], rho[i, j, 1]
                            br, bi = rho[i ^ flip, j, 0], rho[i ^ flip, j, 1]
                            dr, di = coeff[p, 0, 0], coeff[p, 0, 1]
                            er, ei = coeff[p, 1, 0], coeff[p, 1, 1]
                            work[i, j, 0] = dr * ar - di * ai + er * br - ei * bi
                            work[i, j, 1] = dr * ai + di * ar + er * bi + ei * br
                        T.sync_threads()
                        for i, j in T.Parallel(dim, dim):
                            p = ((j >> a) ^ (j >> b)) & 1
                            ar, ai = work[i, j, 0], work[i, j, 1]
                            br, bi = work[i, j ^ flip, 0], work[i, j ^ flip, 1]
                            dr, di = coeff[p, 0, 0], coeff[p, 0, 1]
                            er, ei = coeff[p, 1, 0], coeff[p, 1, 1]
                            rho[i, j, 0] = dr * ar + di * ai + er * br + ei * bi
                            rho[i, j, 1] = dr * ai - di * ar + er * bi - ei * br
                    else:
                        if kind == 2:
                            for k in T.Parallel(dim // 2):
                                lo = (k & (bit - 1)) | ((k >> a) << (a + 1))
                                hi = lo | bit
                                z[0, k] = rho[lo, lo, 0] - rho[hi, hi, 0]
                                x[0, k] = 2.0 * rho[lo, hi, 0]
                                y[0, k] = -2.0 * rho[lo, hi, 1]
                            T.reduce_sum(z, rz, dim=1)
                            T.reduce_sum(x, rx, dim=1)
                            T.reduce_sum(y, ry, dim=1)
                            result[row, b, 0], result[row, b, 1], result[row, b, 2] = rz[0], rx[0], ry[0]
                        for i, j in T.Parallel(dim, dim):
                            if kind == 0:
                                ii, jj = (i >> a) & 1, (j >> a) & 1
                                ar, ai = rho[i & ~bit, j & ~bit, 0], rho[i & ~bit, j & ~bit, 1]
                                br, bi = initial[b, ii, jj, 0], initial[b, ii, jj, 1]
                                work[i, j, 0] = ar * br - ai * bi
                                work[i, j, 1] = ar * bi + ai * br
                            else:
                                for c in T.unroll(2):
                                    work[i, j, c] = T.if_then_else(
                                        ((i | j) & bit) == 0, rho[i, j, c] + rho[i | bit, j | bit, c], 0.0)
                        T.sync_threads()
                        for i, j in T.Parallel(dim, dim):
                            rho[i, j, 0], rho[i, j, 1] = work[i, j, 0], work[i, j, 1]
                    T.sync_threads()
        return kernel

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frontier().export_library(str(directory / 'readout.so'))
    (directory / 'metadata.json').write_text(json.dumps(dict(batch=batch, threads=threads)))


def frontier_shape(model, plan):
    width, (_, columns, _, actions) = plan
    return model.q_enc, model.num_features, width, len(columns), len(actions)


class FrontierSweepReadout:
    def __init__(self, model, directory):
        from pqfmlib.experimental.sih_topology_sweep import _PrecompiledLocalReadout

        from pqfmlib.experimental.sih_wide_frontier_tilelang import live_bases

        metadata = json.loads((Path(directory) / 'metadata.json').read_text())
        dim = metadata.get('workspace')
        compact = metadata.get('compact', False)
        plan = frontier_plan(model, max_width=7 if compact else dim.bit_length()-1 if dim else 5,
                             compact=compact)
        if plan is None:
            raise ValueError('Circuit has no supported exact density frontier')
        width, inputs = plan
        if dim:
            if dim != 1 << width:
                raise ValueError('Precompiled frontier width does not match the circuit')
            inputs += live_bases(width, inputs[-1])
        self.reader = _PrecompiledLocalReadout(model, directory, inputs)

    def transform(self, values):
        return self.reader.transform(values)
