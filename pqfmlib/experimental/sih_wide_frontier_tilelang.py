"""Exact TileLang frontier with in-place parity blocks and compact live bases."""

import json
from pathlib import Path

import numpy as np


def live_bases(width, actions):
    """List pivot states with the operation's bit zero and all vacant bits zero."""
    masks = []
    sizes = []
    pivots = []
    live = 0
    basis = np.arange(1 << width, dtype=np.int32)
    for kind, a, b, _ in actions:
        bit = 1 << int(a)
        if kind == 0:
            live |= bit
        mask = live & ~bit
        masks.append(mask)
        if kind == 2:
            live &= ~bit
    unique = {mask: idx for idx, mask in enumerate(dict.fromkeys(masks))}
    for mask, idx in unique.items():
        states = basis[basis & ~mask == 0]
        sizes.append(len(states))
        pivots.append(np.pad(states, (0, len(basis) // 2 - len(states))))
    return np.array([unique[mask] for mask in masks], np.int32), np.array(sizes, np.int32), np.stack(pivots)


def build_wide_frontier(directory, qubits, features, width, groups, steps, nbases, *,
                        batch=128, threads=128, compact=False, exchange=False):
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
                   step_basis: T.Tensor((steps,), 'int32'),
                   basis_sizes: T.Tensor((nbases,), 'int32'),
                   bases: T.Tensor((nbases, dim // 2), 'int32'),
                   rho: T.Tensor((batch, dim, dim, 2), 'float64'),
                   result: T.Tensor((batch, qubits, 3), 'float64')):
            with T.Kernel(batch, threads=threads) as row:
                coefficients = T.alloc_shared((2, 2, 2), 'float64')
                z = T.alloc_fragment((1, dim // 2), 'float64')
                x = T.alloc_fragment((1, dim // 2), 'float64')
                y = T.alloc_fragment((1, dim // 2), 'float64')
                rz = T.alloc_fragment((1,), 'float64')
                rx = T.alloc_fragment((1,), 'float64')
                ry = T.alloc_fragment((1,), 'float64')
                block = T.alloc_local((2, 2, 2), 'float64')
                mixed = T.alloc_local((2, 2, 2), 'float64')
                lane = T.get_thread_binding()
                if lane == 0:
                    rho[row, 0, 0, 0], rho[row, 0, 0, 1] = 1.0, 0.0
                T.sync_threads()
                for step in T.serial(steps):
                    kind, a, b, edge = actions[step, 0], actions[step, 1], actions[step, 2], actions[step, 3]
                    bit = 1 << a
                    basis_id = step_basis[step]
                    size = basis_sizes[basis_id]
                    log_size = 31 - T.clz(T.cast(size, 'uint32'))
                    if kind == 1:
                        flip = bit | (1 << b)
                        if exchange:
                            if lane == 0:
                                theta = T.float64(6.283185307179586) * T.tanh(values[row, columns[edge, 0]] / 3.0) * scales[edge, 0]
                                c, s = T.cos(theta), T.sin(theta)
                                coefficients[1, 0, 0], coefficients[1, 0, 1] = c*c, c*s
                                coefficients[1, 1, 0], coefficients[1, 1, 1] = s*s, -c*s
                                coefficients[0, 0, 0], coefficients[0, 0, 1] = 1.0, 0.0
                                coefficients[0, 1, 0], coefficients[0, 1, 1] = 0.0, 0.0
                        else:
                            if lane < 2:
                                ax = T.float64(3.141592653589793) * T.tanh(values[row, columns[edge, 0]] / 3.0) * scales[edge, 0]
                                ay = T.float64(3.141592653589793) * T.tanh(values[row, columns[edge, 1]] / 3.0) * scales[edge, 1]
                                az = T.float64(3.141592653589793) * T.tanh(values[row, columns[edge, 2]] / 3.0) * scales[edge, 2]
                                angle = T.if_then_else(lane == 0, ax - ay, ax + ay)
                                cr, ci = T.cos(az), T.if_then_else(lane == 0, -T.sin(az), T.sin(az))
                                ca, sa = T.cos(angle), T.sin(angle)
                                coefficients[lane, 0, 0], coefficients[lane, 0, 1] = cr * ca, ci * ca
                                coefficients[lane, 1, 0], coefficients[lane, 1, 1] = ci * sa, -cr * sa
                        T.sync_threads()
                        for chunk in T.serial(T.ceildiv(size * size, threads)):
                            index = chunk * threads + lane
                            if index < size * size:
                                i = bases[basis_id, index >> log_size]
                                j = bases[basis_id, index & (size - 1)]
                                pi, pj = (i >> b) & 1, (j >> b) & 1
                                if not exchange or (pi | pj) != 0:
                                    for uv_c in T.unroll(8):
                                        u, v, c = uv_c // 4, uv_c // 2 % 2, uv_c % 2
                                        block[u, v, c] = rho[row, i ^ (u * flip), j ^ (v * flip), c]
                                    for uv in T.unroll(4):
                                        u, v = uv // 2, uv % 2
                                        ar, ai = block[u, v, 0], block[u, v, 1]
                                        br, bi = block[1-u, v, 0], block[1-u, v, 1]
                                        dr, di = coefficients[pi, 0, 0], coefficients[pi, 0, 1]
                                        er, ei = coefficients[pi, 1, 0], coefficients[pi, 1, 1]
                                        mixed[u, v, 0] = dr*ar - di*ai + er*br - ei*bi
                                        mixed[u, v, 1] = dr*ai + di*ar + er*bi + ei*br
                                    for uv in T.unroll(4):
                                        u, v = uv // 2, uv % 2
                                        ar, ai = mixed[u, v, 0], mixed[u, v, 1]
                                        br, bi = mixed[u, 1-v, 0], mixed[u, 1-v, 1]
                                        dr, di = coefficients[pj, 0, 0], coefficients[pj, 0, 1]
                                        er, ei = coefficients[pj, 1, 0], coefficients[pj, 1, 1]
                                        ri, rj = i ^ (u * flip), j ^ (v * flip)
                                        real = dr*ar + di*ai + er*br + ei*bi
                                        imag = dr*ai - di*ar + er*bi - ei*br
                                        rho[row, ri, rj, 0] = real
                                        rho[row, ri, rj, 1] = imag
                    else:
                        if kind == 2:
                            for k in T.Parallel(dim // 2):
                                z[0, k], x[0, k], y[0, k] = 0.0, 0.0, 0.0
                                if k < size:
                                    lo = bases[basis_id, k]
                                    hi = lo | bit
                                    z[0, k] = rho[row, lo, lo, 0] - rho[row, hi, hi, 0]
                                    x[0, k] = 2.0 * rho[row, lo, hi, 0]
                                    y[0, k] = -2.0 * rho[row, lo, hi, 1]
                            T.reduce_sum(z, rz, dim=1)
                            T.reduce_sum(x, rx, dim=1)
                            T.reduce_sum(y, ry, dim=1)
                            result[row, b, 0], result[row, b, 1], result[row, b, 2] = rz[0], rx[0], ry[0]
                            T.sync_threads()
                        for chunk in T.serial(T.ceildiv(size * size, threads)):
                            index = chunk * threads + lane
                            if index < size * size:
                                i = bases[basis_id, index >> log_size]
                                j = bases[basis_id, index & (size - 1)]
                                if kind == 0:
                                    ar, ai = rho[row, i, j, 0], rho[row, i, j, 1]
                                    for uv in T.unroll(4):
                                        u, v = uv // 2, uv % 2
                                        br, bi = initial[b, u, v, 0], initial[b, u, v, 1]
                                        rho[row, i | (u*bit), j | (v*bit), 0] = ar*br - ai*bi
                                        rho[row, i | (u*bit), j | (v*bit), 1] = ar*bi + ai*br
                                else:
                                    for c in T.unroll(2):
                                        rho[row, i, j, c] += rho[row, i | bit, j | bit, c]
                    T.sync_threads()
        return kernel

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frontier().export_library(str(directory / 'readout.so'))
    (directory / 'metadata.json').write_text(json.dumps(dict(batch=batch, threads=threads, workspace=dim, compact=compact, exchange=exchange, bases=nbases)))


def wide_frontier_shape(model, plan):
    from pqfmlib.experimental.sih_frontier_tilelang import frontier_shape

    return (*frontier_shape(model, plan), len(live_bases(plan[0], plan[1][-1])[1]))
