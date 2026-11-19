"""Tied-exchange transfer in charge eigenoperators, with conjugate reuse."""

import json
from pathlib import Path

import numpy as np
from pqfmlib.experimental.sih_frontier_tilelang import frontier_plan
from pqfmlib.experimental.sih_topology_sweep import _PrecompiledLocalReadout

# Local pair coefficients in the basis I, Z, (X+iY)/sqrt(2), (X-iY)/sqrt(2).
_GROUPS = np.array(
    [
        [[1, 0], [0, 1], [2, 3], [3, 2]],
        [[2, 0], [0, 2], [1, 2], [2, 1]],
        [[3, 0], [0, 3], [1, 3], [3, 1]],
    ],
    dtype=np.int32,
)
_TYPES = np.full((4, 4), -1, np.int32)
for j, pairs in enumerate(_GROUPS):
    _TYPES[pairs[:, 0], pairs[:, 1]] = j


def adjoint(word):
    return word ^ ((word >> 1) & 0x55555555)


def ladder_plan(width, actions, aligned):
    dim = 1 << (2 * width)
    words = np.arange(dim, dtype=np.int32)
    dagger = adjoint(words)
    canonical = np.minimum(words, dagger)
    required = np.zeros(dim, dtype=bool)
    plan = []
    for kind, a, b, _ in actions[::-1]:
        aa, bb = 2 * int(a), 2 * int(b)
        if kind == 2:
            required[np.array([1, 2], np.int32) << aa] = True
            row = np.empty((0, 5), np.int32)
        elif kind == 0:
            demand = words[required & ((words >> aa & 3) != 0)]
            base = demand & ~(3 << aa)
            required[demand] = False
            digit = demand >> aa & 3
            keep = (digit == 1) if int(b) == aligned else np.ones(len(base), bool)
            required[canonical[base[keep]]] = True
            parents, reverse = np.unique(canonical[base], return_inverse=True)
            is_conjugate = base != canonical[base]
            local = np.where(is_conjugate & (digit >= 2), digit ^ 1, digit)
            row = np.zeros((len(parents), 5), np.int32)
            row[:, 0] = parents
            row[reverse, local] = np.where(is_conjugate, -(demand + 1), demand + 1)
        else:
            demand = words[required]
            groups = _TYPES[demand >> aa & 3, demand >> bb & 3]
            change = groups >= 0
            demand, groups = demand[change], groups[change]
            base = demand & ~((3 << aa) | (3 << bb))
            direct = base * 4 + groups
            reverse = dagger[base] * 4 + np.where(groups == 0, 0, 3 - groups)
            keys = np.unique(np.minimum(direct, reverse))
            groups, base = keys & 3, keys >> 2
            local = _GROUPS[groups]
            operands = base[:, None] | (local[:, :, 0] << aa) | (local[:, :, 1] << bb)
            required[canonical[operands]] = True
            # Signed-index encoding carries the conjugacy bit without a branch.
            operands = np.where(
                operands == canonical[operands], operands, ~canonical[operands]
            )
            row = np.column_stack((operands, groups))
        plan.append(row)
    assert np.array_equal(words[required], [0])
    offsets = np.r_[
        np.int32(0), np.cumsum([len(row) for row in plan[::-1]], dtype=np.int32)
    ]
    return offsets, np.concatenate(plan[::-1])


def aligned_plan(width, actions):
    choices = []
    for wire in sorted(actions[actions[:, 0] == 0, 2]):
        offsets, work = ladder_plan(width, actions, wire)
        counts = np.diff(offsets)
        score = 4 * sum(counts[actions[:, 0] == 1]) + len(work)
        choices.append((score, wire, offsets, work))
    return min(choices, key=lambda choice: choice[0])[1:]


class NativeLadderReadout(_PrecompiledLocalReadout):
    """Read local Z, X and Y from an offline isotropic exchange kernel."""

    def __init__(self, model, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        plan = frontier_plan(model, max_width=metadata["shape"][2], compact=True)
        if model.topology != "native" or not model.use_tanh_scaling or plan is None:
            raise ValueError(
                "Ladder reader requires a native circuit with tanh angle scaling within the precompiled frontier width"
            )
        width, (initial, columns, scales, actions) = plan
        shape = [model.q_enc, model.num_features, width, len(columns), len(actions)]
        if metadata["shape"] != shape:
            raise ValueError(
                f'Precompiled ladder shape {metadata["shape"]} does not match circuit shape {shape}'
            )
        if np.any(columns != columns[:, :1]) or np.any(scales != scales[:, :1]):
            raise ValueError("Ladder reader requires tied exchange")
        aligned, offsets, work = aligned_plan(width, actions)
        if len(work) > metadata["entries"]:
            raise ValueError("Precompiled ladder has too few coefficient addresses")
        bloch = np.column_stack(
            (
                2 * initial[:, 0, 1, 0],
                -2 * initial[:, 0, 1, 1],
                initial[:, 0, 0, 0] - initial[:, 1, 1, 0],
            )
        )
        z = bloch[aligned] / np.linalg.norm(bloch[aligned])
        x = np.cross(np.eye(3)[np.argmin(np.abs(z))], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        frame = np.column_stack((x, y, z))
        transformed = bloch @ frame
        # A common rotation commutes with every isotropic exchange. In
        # this frame the chosen input has exactly zero ladder components.
        transformed[aligned, :2] = 0
        initial_bloch = np.column_stack(
            (
                transformed[:, 2],
                transformed[:, 0] / np.sqrt(2),
                transformed[:, 1] / np.sqrt(2),
            )
        )
        super().__init__(
            model,
            directory,
            (
                initial_bloch,
                frame,
                columns[:, 0].copy(),
                scales[:, 0].copy(),
                actions,
                offsets,
                np.pad(work, ((0, metadata["entries"] - len(work)), (0, 0))),
            ),
        )
        cohort = metadata["cohort"]
        scratch = self.torch.empty(
            (self.batch // cohort, 1 << (2 * width), 2, cohort),
            dtype=self.torch.float64,
            device="cuda",
        )
        self.plan = (*self.plan, scratch)
        tail_dir = directory / "tail"
        self.tail = NativeLadderReadout(model, tail_dir) if tail_dir.is_dir() else None

    def transform(self, values):
        tail_rows = len(values) % self.batch
        if self.tail is None or not 0 < tail_rows <= self.tail.batch:
            return super().transform(values)
        if tail_rows == len(values):
            return self.tail.transform(values)
        return np.concatenate(
            (
                super().transform(values[:-tail_rows]),
                self.tail.transform(values[-tail_rows:]),
            )
        )


def build_native_ladder(
    directory,
    qubits,
    features,
    width,
    groups,
    steps,
    entries,
    *,
    batch,
    threads,
    cohort,
    precompute=False,
):
    import tilelang
    import tilelang.language as T

    dim = 1 << (2 * width)
    workers = threads // cohort
    coefficient_count = groups if precompute else 1
    assert threads % cohort == batch % cohort == 0

    @T.macro
    def get_component(rho, group, sample, encoded, part, block, index):
        address = T.if_then_else(encoded >= 0, encoded, ~encoded)
        block[index, part] = rho[group, address, part, sample] * T.if_then_else(
            part == 1 and encoded < 0, -1.0, 1.0
        )

    @tilelang.jit(target="cuda", execution_backend="tvm_ffi")
    def compile_kernel():
        @T.prim_func
        def kernel(
            values: T.Tensor((batch, features), "float64"),
            bloch: T.Tensor((qubits, 3), "float64"),
            frame: T.Tensor((3, 3), "float64"),
            columns: T.Tensor((groups,), "int32"),
            scales: T.Tensor((groups,), "float64"),
            actions: T.Tensor((steps, 4), "int32"),
            offsets: T.Tensor((steps + 1,), "int32"),
            work: T.Tensor((entries, 5), "int32"),
            rho: T.Tensor((batch // cohort, dim, 2, cohort), "float64"),
            result: T.Tensor((batch, qubits, 3), "float64"),
        ):
            with T.Kernel(batch // cohort, threads=threads) as group:
                coeff = T.alloc_shared((coefficient_count, cohort, 2), "float64")
                block = T.alloc_local((4, 2), "float64")
                output = T.alloc_local((4, 2), "float64")
                addresses = T.alloc_local((4,), "int32")
                tid = T.get_thread_binding()
                lane = tid // cohort
                sample = tid % cohort
                row = group * cohort + sample
                if lane == 0:
                    rho[group, 0, 0, sample] = 1.0
                    rho[group, 0, 1, sample] = 0.0
                if precompute:
                    for chunk in T.serial(T.ceildiv(groups, workers)):
                        gate = chunk * workers + lane
                        if gate < groups:
                            theta = (
                                T.float64(6.283185307179586)
                                * T.tanh(values[row, columns[gate]] / 3.0)
                                * scales[gate]
                            )
                            c, sine = T.cos(theta), T.sin(theta)
                            coeff[gate, sample, 0], coeff[gate, sample, 1] = (
                                sine * sine,
                                c * sine,
                            )
                T.sync_threads()
                for step in T.serial(steps):
                    kind, a, b, edge = (
                        actions[step, 0],
                        actions[step, 1],
                        actions[step, 2],
                        actions[step, 3],
                    )
                    shift_a = 2 * a
                    first = offsets[step]
                    count = offsets[step + 1] - first
                    if kind == 2:
                        if lane == 0:
                            z = rho[group, 1 << shift_a, 0, sample]
                            x = (
                                T.float64(1.4142135623730951)
                                * rho[group, 2 << shift_a, 0, sample]
                            )
                            y = (
                                T.float64(1.4142135623730951)
                                * rho[group, 2 << shift_a, 1, sample]
                            )
                            for k in T.unroll(3):
                                axis = (k + 2) % 3
                                result[row, b, k] = (
                                    frame[axis, 0] * x
                                    + frame[axis, 1] * y
                                    + frame[axis, 2] * z
                                )
                    elif kind == 0:
                        for chunk in T.serial(T.ceildiv(count, workers)):
                            i = chunk * workers + lane
                            if i < count:
                                base = work[first + i, 0]
                                ar, ai = (
                                    rho[group, base, 0, sample],
                                    rho[group, base, 1, sample],
                                )
                                for k_local in T.unroll(3):
                                    k = k_local + 1
                                    dest = work[first + i, k]
                                    if dest != 0:
                                        ptr = T.abs(dest) - 1
                                        br = (
                                            bloch[b, 0] if k_local == 0 else bloch[b, 1]
                                        )
                                        bi = (
                                            0.0
                                            if k_local == 0
                                            else (
                                                bloch[b, 2]
                                                if k_local == 1
                                                else -bloch[b, 2]
                                            )
                                        )
                                        if k_local != 0 and br == 0.0 and bi == 0.0:
                                            rho[group, ptr, 0, sample] = 0.0
                                            rho[group, ptr, 1, sample] = 0.0
                                        else:
                                            rho[group, ptr, 0, sample] = (
                                                ar * br - ai * bi
                                            )
                                            imag = ar * bi + ai * br
                                            rho[group, ptr, 1, sample] = T.if_then_else(
                                                dest > 0, imag, -imag
                                            )
                    else:
                        if not precompute:
                            if lane == 0:
                                theta = (
                                    T.float64(6.283185307179586)
                                    * T.tanh(values[row, columns[edge]] / 3.0)
                                    * scales[edge]
                                )
                                c, sine = T.cos(theta), T.sin(theta)
                                coeff[0, sample, 0], coeff[0, sample, 1] = (
                                    sine * sine,
                                    c * sine,
                                )
                            T.sync_threads()
                        index = edge if precompute else 0
                        ss = coeff[index, sample, 0]
                        for chunk in T.serial(T.ceildiv(count, workers)):
                            i = chunk * workers + lane
                            if i < count:
                                k = work[first + i, 4]
                                cs = T.if_then_else(
                                    k == 2,
                                    -coeff[index, sample, 1],
                                    coeff[index, sample, 1],
                                )
                                for j in T.unroll(4):
                                    addresses[j] = work[first + i, j]
                                    for part in T.unroll(2):
                                        get_component(
                                            rho,
                                            group,
                                            sample,
                                            addresses[j],
                                            part,
                                            block,
                                            j,
                                        )
                                for part in T.unroll(2):
                                    dp = block[0, part] - block[1, part]
                                    dr = block[2, part] - block[3, part]
                                    dx = block[0, 1 - part] - block[1, 1 - part]
                                    dy = block[2, 1 - part] - block[3, 1 - part]
                                    sign = T.if_then_else(part == 0, cs, -cs)
                                    u = -ss * dp + sign * dy
                                    v = -ss * dr + sign * dx
                                    output[0, part], output[1, part] = (
                                        block[0, part] + u,
                                        block[1, part] - u,
                                    )
                                    output[2, part], output[3, part] = (
                                        block[2, part] + v,
                                        block[3, part] - v,
                                    )
                                for j in T.unroll(4):
                                    loc = addresses[j]
                                    # A self-adjoint group contains both orientations
                                    # of one word. Only its canonical output writes.
                                    ptr = T.if_then_else(loc >= 0, loc, ~loc)
                                    paired = T.if_then_else(
                                        j == 2, addresses[3], addresses[2]
                                    )
                                    if loc >= 0 or k != 0 or j < 2 or paired != ptr:
                                        rho[group, ptr, 0, sample] = output[j, 0]
                                        rho[group, ptr, 1, sample] = T.if_then_else(
                                            loc >= 0, output[j, 1], -output[j, 1]
                                        )
                    T.sync_threads()

        return kernel

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    compile_kernel().export_library(str(directory / "readout.so"))
    (directory / "metadata.json").write_text(
        json.dumps(
            dict(
                batch=batch,
                threads=threads,
                cohort=cohort,
                entries=entries,
                shape=[qubits, features, width, groups, steps],
            )
        )
    )
