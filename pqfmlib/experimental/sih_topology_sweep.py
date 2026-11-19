"""Offline SIH-SUS topology profiles, circuit exports, and exact TileLang simulation."""

from pathlib import Path
from contextlib import contextmanager
from threading import Lock
import csv
import fcntl
import json
import os

import numpy as np


@contextmanager
def locked_sweep_journal(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = {}
        with path.open('a+b') as journal:
            journal.seek(0)
            while True:
                offset = journal.tell()
                line = journal.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    journal.truncate(offset)
                    journal.flush()
                    os.fsync(journal.fileno())
                    break
                record = json.loads(line)
                name = record['name']
                if name in records:
                    raise ValueError(f'Duplicate sweep case {name} in {path}')
                records[name] = record
        yield records


def topology_profiles(root, *, complete=False):
    """Preserve 50 PCA inputs and cancel the block-dependent angle normalization."""
    profiles = {}
    for qubits in (19, 26):
        for topology in ('chain', 'plaquettes', 'full', 'native'):
            layouts = ('auto', 'fixed') if topology == 'native' else ('auto',)
            for layout in layouts:
                for label in ('axes', 'balanced', 'shared'):
                    strengths = (.05, .1, .2) if complete or label == 'balanced' else (.1,)
                    for strength in strengths:
                        name = f'q{qubits}_{topology}_{layout}_{label}_s{round(100*strength):02d}'
                        config = dict(q_enc=qubits, n_components=50, topology=topology,
                                      encoding_mode='shared_feature' if label == 'shared' else 'multi_axis',
                                      balanced_encoding=label == 'balanced', ideal=topology != 'native')
                        if topology == 'native':
                            config.update(topology_file=str(root / 'graphs' / 'sherbrooke.json'),
                                          use_fixed_phys_nodes=layout == 'fixed')
                            if layout == 'fixed':
                                config['fixed_phys_nodes_file'] = str(root / 'graphs' / f'fixed_q{qubits}.json')
                        profiles[name] = config, strength
    return profiles


PAPER_SWEEP_STRENGTHS = (.025, .075, .15, .40)


def extend_topology_manifest(manifest):
    """Add angle scales without changing geometry, kernel shape, or prior case IDs."""
    reference = {name.rsplit('_', 1)[0]: entry for name, entry in manifest.items()
                 if entry['strength'] == .10}
    if len(manifest) != 90 or len(reference) != 30:
        raise ValueError('The paper sweep requires the complete 90-profile manifest')
    extra = {
        f'{geometry}_s{round(1000 * strength):03d}':
            entry | dict(config=entry['config'] | dict(alpha=strength * entry['blocks']), strength=strength)
        for geometry, entry in reference.items()
        for strength in PAPER_SWEEP_STRENGTHS
    }
    if extra.keys() & manifest.keys():
        raise ValueError('The paper sweep overlaps existing profile IDs')
    return extra | manifest


def routing_diagnostics(y, classical, hybrid, router):
    """Evaluate fixed routing policies from the same held-out predictions."""
    from sklearn.metrics import auc, average_precision_score, precision_recall_curve

    def metrics(score):
        precision, recall, _ = precision_recall_curve(y, score)
        return dict(ap=float(average_precision_score(y, score)), auprc=float(auc(recall, precision)))

    hard = {}
    for threshold in (.5, .6, .7, .8, .9):
        use_hybrid = router > threshold
        hard[str(threshold)] = metrics(np.where(use_hybrid, hybrid, classical)) | dict(
            routing_fraction=float(use_hybrid.mean()))
    blends = {str(weight): metrics((1 - weight) * classical + weight * hybrid)
              for weight in (.05, .10, .25, .50)}
    return dict(hard=hard, fixed_blends=blends)


def write_native_graph(root, factory, base):
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

    from pqfmlib.hardware.topology import save_topology_snapshot
    from pqfmlib.utils.io import save_json

    graph = root / 'graphs' / 'sherbrooke.json'
    save_topology_snapshot(FakeSherbrooke(), graph)
    for qubits in (19, 26):
        model = factory(base | dict(q_enc=qubits, topology='native', ideal=False,
                                    topology_file=str(graph))).fit(np.zeros((1, 50)))
        save_json(list(reversed(model.phys_nodes)), graph.parent / f'fixed_q{qubits}.json')


def active_circuit(model):
    """Bind structural zero slots while keeping data-bearing parameters symbolic."""
    from qiskit import QuantumCircuit

    prepared = model._prepared_execution
    probe, _ = model._make_theta_matrix(np.linspace(.1, 1, model.num_features)[None])
    fixed = {p: 0. for p, value in zip(prepared.param_order, probe[0]) if value == 0}
    circuit = prepared.qc_t.assign_parameters(fixed)
    active = QuantumCircuit(circuit.num_qubits)
    for instruction in circuit.data:
        op = instruction.operation
        if op.name == 'barrier' or (op.name in ('rxx', 'ryy', 'rzz') and op.params[0] == 0):
            continue
        active.append(op, [circuit.find_bit(q).index for q in instruction.qubits])
    return active


def write_drawings(root, name, model, active):
    from qiskit import qpy
    import matplotlib.pyplot as plt

    root.mkdir(parents=True, exist_ok=True)
    logical = model._prepared_execution.qc_t
    for label, circuit in [('logical', logical), ('active', active)]:
        with (root / f'{label}.qpy').open('wb') as stream:
            qpy.dump(circuit, stream)
        (root / f'{label}.txt').write_text(str(circuit.draw('text', fold=120)))
    fig = active.draw('mpl', fold=-1, scale=.42)
    fig.savefig(root / 'active.svg', bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 6))
    xy = np.column_stack([np.cos(np.linspace(0, 2*np.pi, model.q_enc, endpoint=False)),
                          np.sin(np.linspace(0, 2*np.pi, model.q_enc, endpoint=False))])
    for i, j in model.edges_act:
        ax.plot(*xy[[i, j]].T, color='#c4cbd1', lw=.6)
    active_edges = {tuple(active.find_bit(q).index for q in inst.qubits)
                    for inst in active.data if len(inst.qubits) == 2}
    for i, j in active_edges:
        ax.plot(*xy[[i, j]].T, color='#176b95', lw=1.2)
    ax.scatter(*xy.T, s=180, c='#edf6ff', edgecolors='#176b95', zorder=3)
    for i, (x, y) in enumerate(xy):
        ax.text(x, y, str(i), ha='center', va='center', fontsize=8, zorder=4)
    ax.set(title=f'{name}\nBlue edges carry data, grey edges have zero angles', aspect='equal')
    ax.axis('off')
    fig.savefig(root / 'topology.svg', bbox_inches='tight')
    plt.close(fig)


def prepare_topology_sweep(root, factory, base, draw=False, *, complete=False):
    from pqfmlib.experimental.heisenberg_tilelang import causal_cones

    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if not (root / 'graphs' / 'sherbrooke.json').exists():
        write_native_graph(root, factory, base)
    manifest = {}
    profiles = topology_profiles(root, complete=complete)
    for name, (config, strength) in profiles.items():
        probe = np.zeros((1, config['n_components']))
        model = factory(base | config).fit(probe)
        blocks = model.theta_info['num_blocks']
        config['alpha'] = strength * blocks
        model = factory(base | config).fit(probe)
        active = active_circuit(model)
        gates = [([active.find_bit(q).index for q in inst.qubits], None, None)
                 for inst in active.data]
        *_, width = causal_cones(gates, model.q_enc)
        rotations = [inst.operation.name for inst in active.data if len(inst.qubits) == 2]
        backend = ('commuting_x' if rotations and set(rotations) == {'rxx'}
                   else 'aot' if width <= 8 else 'forest' if forest_plan(model) is not None else 'generic_tilelang')
        manifest[name] = dict(config=config, strength=strength, blocks=blocks,
                              edges=len(model.edges_act), active_rotations=len(active.data)-model.q_enc,
                              width=width, backend=backend, physical_nodes=model.phys_nodes)
        if draw and strength == .1:
            write_drawings(root / 'circuits' / name, name, model, active)
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    with (root / 'summary.csv').open('w', newline='') as stream:
        keys = ('strength', 'blocks', 'edges', 'active_rotations', 'width', 'backend')
        writer = csv.writer(stream)
        writer.writerow(('config', *keys))
        writer.writerows((name, *(entry[k] for k in keys)) for name, entry in manifest.items())
    return manifest


def _input_columns(model):
    theta, _ = model._make_theta_matrix(np.linspace(.1, 1., model.num_features)[None])
    active_slots = np.flatnonzero(theta[0])
    input_order = active_slots[np.argsort(theta[0, active_slots])]
    return {model._prepared_execution.param_order[slot]: column for column, slot in enumerate(input_order)}


def commuting_x_plan(model):
    """Prepare one input column and neighbor Bloch coordinate for each incident XX."""
    from qiskit.quantum_info import Operator

    circuit = active_circuit(model)
    mapping = _input_columns(model)
    bloch = np.empty((model.q_enc, 3), dtype=np.float64)
    edges = [[] for _ in range(model.q_enc)]
    for item in circuit.data:
        sites = [circuit.find_bit(q).index for q in item.qubits]
        if item.operation.name == 'unitary' and len(sites) == 1:
            a, b = Operator(item.operation).data[:, 0]
            cross = 2*a.conjugate()*b
            bloch[sites[0]] = abs(a)**2-abs(b)**2, cross.real, cross.imag
        elif item.operation.name == 'rxx':
            angle = item.operation.params[0]
            parameter, = angle.parameters
            if float(angle.bind({parameter: 0})) != 0:
                raise ValueError('Expected an XX rotation with a scaled input.')
            scale = float(angle.gradient(parameter))
            left, right = sites
            edges[left].append((mapping[parameter], scale, right))
            edges[right].append((mapping[parameter], scale, left))
        else:
            raise ValueError('Commuting XX readout requires local preparation and XX rotations.')
    degree = max(map(len, edges))
    columns = np.zeros((model.q_enc, degree), dtype=np.int32)
    scales = np.zeros((model.q_enc, degree))
    neighbors = np.zeros((model.q_enc, degree))
    for site, incident in enumerate(edges):
        for j, (column, scale, neighbor) in enumerate(incident):
            columns[site, j], scales[site, j], neighbors[site, j] = column, scale, bloch[neighbor, 1]
    return bloch, columns, scales, neighbors


def build_commuting_x(directory, qubits, features, degree, *, batch, threads):
    """Export a fixed-batch kernel. Each lane evaluates one sample/site pair."""
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target='cuda', execution_backend='tvm_ffi')
    def kernel():
        @T.prim_func
        def readout(x: T.Tensor((batch, features), 'float64'),
                    bloch: T.Tensor((qubits, 3), 'float64'),
                    columns: T.Tensor((qubits, degree), 'int32'),
                    scales: T.Tensor((qubits, degree), 'float64'),
                    neighbors: T.Tensor((qubits, degree), 'float64'),
                    result: T.Tensor((batch, qubits, 3), 'float64')):
            with T.Kernel(T.ceildiv(batch * qubits, threads), threads=threads) as block:
                for lane in T.Parallel(threads):
                    item = block * threads + lane
                    if item < batch * qubits:
                        row, q = item // qubits, item % qubits
                        c = T.alloc_var('float64', init=1.)
                        s = T.alloc_var('float64', init=0.)
                        for j in T.serial(degree):
                            angle = 6.283185307179586 * T.tanh(x[row, columns[q, j]] / 3.) * scales[q, j]
                            re = T.cos(angle)
                            im = neighbors[q, j] * T.sin(angle)
                            next_c = c * re - s * im
                            s = s * re + c * im
                            c = next_c
                        result[row, q, 0] = bloch[q, 0] * c + bloch[q, 2] * s
                        result[row, q, 1] = bloch[q, 1]
                        result[row, q, 2] = bloch[q, 2] * c - bloch[q, 0] * s
        return readout

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    kernel().export_library(str(directory / 'readout.so'))
    (directory / 'metadata.json').write_text(json.dumps(dict(batch=batch, threads=threads)))


class _PrecompiledLocalReadout:
    def __init__(self, model, directory, plan):
        import torch

        directory = Path(directory)
        self.torch = torch
        self.plan = tuple(torch.as_tensor(x, device='cuda') for x in plan)
        metadata = json.loads((directory / 'metadata.json').read_text())
        self.batch = metadata['batch']
        if dim := metadata.get('workspace'):
            self.plan += (torch.empty((self.batch, dim, dim, 2), dtype=torch.float64, device='cuda'),)
        self.evaluate = _load_offline_kernel(directory / 'readout.so')
        self.input = torch.zeros((self.batch, model.num_features), dtype=torch.float64, device='cuda')
        self.result = torch.empty((self.batch, model.q_enc, 3), dtype=torch.float64, device='cuda')

    def transform(self, values):
        torch = self.torch
        values = torch.as_tensor(np.ascontiguousarray(values, dtype=np.float64), device='cuda')
        result = torch.empty((len(values), self.result.shape[1]*3), dtype=torch.float64, device='cuda')
        for start in range(0, len(values), self.batch):
            rows = values[start:start + self.batch]
            self.input[:len(rows)].copy_(rows)
            self.evaluate(self.input, *self.plan, self.result)
            result[start:start + len(rows)].copy_(self.result[:len(rows)].flatten(1))
        return result.cpu().numpy()


class CommutingXSweepReadout(_PrecompiledLocalReadout):
    """Exact local observables for commuting XX gates on a product state."""

    def __init__(self, model, directory):
        super().__init__(model, directory, commuting_x_plan(model))


_OFFLINE_KERNEL_LOCK = Lock()
_OFFLINE_KERNELS = {}


class _OfflineKernel:
    def __init__(self, module):
        self.module = module
        self.ready = False

    def __call__(self, *args):
        if not self.ready:
            # TVM resolves the embedded CUDA function lazily on the first launch.
            with _OFFLINE_KERNEL_LOCK:
                if not self.ready:
                    result = self.module(*args)
                    self.ready = True
                    return result
        return self.module(*args)


def _load_offline_kernel(library):
    from tilelang import tvm

    library = library.resolve()
    with _OFFLINE_KERNEL_LOCK:
        if library not in _OFFLINE_KERNELS:
            if not library.is_file():
                raise FileNotFoundError(f'Missing offline TileLang kernel {library}')
            _OFFLINE_KERNELS[library] = _OfflineKernel(tvm.runtime.load_module(str(library)))
        return _OFFLINE_KERNELS[library]


class GenericTileLangSweepReadout:
    """Exact TileLang state evolution after binding structural zero rotations."""

    def __init__(self, model, *, batch_size=1, profile=False, strategy='auto', artifacts=None,
                 matrix_backend=None, parity_gates=False):
        from dataclasses import replace
        from pqfmlib.core.tilelang_execution import TileLangExecutor

        self.model = model
        prepared = model._prepared_execution
        circuit = active_circuit(model)
        self.indices = [i for i, p in enumerate(prepared.param_order) if p in circuit.parameters]
        order = tuple(prepared.param_order[i] for i in self.indices)
        self.prepared = replace(prepared, qc_t=circuit, param_order=order)
        self.offline_batch = batch_size if artifacts is not None else 1
        self.executor = TileLangExecutor(precision='double', batch_size=batch_size, profile=profile,
                                         strategy=strategy, parity_gates=parity_gates,
                                         matrix_backend=matrix_backend or ('auto' if artifacts is None else 'cpu'))
        if artifacts is not None:
            directory = Path(artifacts)

            def load(kernel, *signature):
                key = (kernel, *signature)
                cache = self.executor._kernel_cache
                if key not in cache:
                    library = directory / ('_'.join(map(str, key)) + '.so')
                    cache[key] = _load_offline_kernel(library)
                return cache[key]

            self.executor._launcher = load

    def transform(self, values):
        angles, _ = self.model._make_theta_matrix(values)
        parameters = angles[:, self.indices]
        spare = (-len(parameters)) % self.offline_batch
        if spare:
            parameters = np.pad(parameters, ((0, spare), (0, 0)))
        return self.executor.execute(self.prepared, parameters)[0][:len(values)]


def forest_plan(model):
    """Return active edge groups when each gate joins two previously disjoint components."""
    from itertools import groupby
    from qiskit.quantum_info import Operator

    circuit = active_circuit(model)
    mapping = _input_columns(model)
    bloch = np.empty((model.q_enc, 3), dtype=np.float64)

    def sites(instruction):
        return tuple(circuit.find_bit(q).index for q in instruction.qubits)

    for item in circuit.data[:model.q_enc]:
        a, b = Operator(item.operation).data[:, 0]
        cross = 2*a.conjugate()*b
        bloch[sites(item)[0]] = cross.real, cross.imag, abs(a)**2-abs(b)**2

    components = list(range(model.q_enc))
    edges, columns, scales = [], [], []
    for (left, right), group in groupby(circuit.data[model.q_enc:], key=sites):
        first, second = components[left], components[right]
        if first == second:
            return None
        components = [first if component == second else component for component in components]
        column = np.zeros(3, dtype=np.int32)
        scale = np.zeros(3)
        for item in group:
            axis = 'xyz'.index(item.operation.name[-1])
            angle = item.operation.params[0]
            parameter, = angle.parameters
            if scale[axis] != 0 or float(angle.bind({parameter: 0})) != 0:
                return None
            column[axis], scale[axis] = mapping[parameter], float(angle.gradient(parameter))
        edges.append((left, right))
        columns.append(column)
        scales.append(scale)
    return bloch, np.asarray(edges, dtype=np.int32), np.asarray(columns), np.asarray(scales)


def build_forest(directory, qubits, features, edges, *, batch=1024, threads=128):
    """Export exact one-qubit reductions through a forest of Heisenberg gates."""
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target='cuda', execution_backend='tvm_ffi')
    def kernel():
        @T.prim_func
        def readout(x: T.Tensor((batch, features), 'float64'),
                    bloch: T.Tensor((qubits, 3), 'float64'),
                    sites: T.Tensor((edges, 2), 'int32'),
                    columns: T.Tensor((edges, 3), 'int32'),
                    scales: T.Tensor((edges, 3), 'float64'),
                    result: T.Tensor((batch, qubits, 3), 'float64')):
            with T.Kernel(T.ceildiv(batch, threads), threads=threads) as block:
                for lane in T.Parallel(threads):
                    row = block * threads + lane
                    if row < batch:
                        for q in T.serial(qubits):
                            for axis in T.unroll(3):
                                result[row, q, axis] = bloch[q, axis]
                        for e in T.serial(edges):
                            i, j = sites[e, 0], sites[e, 1]
                            a = T.alloc_local((3,), 'float64')
                            b = T.alloc_local((3,), 'float64')
                            c = T.alloc_local((3,), 'float64')
                            s = T.alloc_local((3,), 'float64')
                            for axis in T.unroll(3):
                                a[axis] = result[row, i, axis]
                                b[axis] = result[row, j, axis]
                                angle = 6.283185307179586 * T.tanh(x[row, columns[e, axis]] / 3.) * scales[e, axis]
                                c[axis], s[axis] = T.cos(angle), T.sin(angle)
                            for axis in T.unroll(3):
                                v, w = (axis+1) % 3, (axis+2) % 3
                                same = c[v]*c[w]
                                swap = s[v]*s[w]
                                plus, minus = s[v]*c[w], c[v]*s[w]
                                result[row, i, axis] = same*a[axis] + swap*b[axis] + plus*a[w]*b[v] - minus*a[v]*b[w]
                                result[row, j, axis] = same*b[axis] + swap*a[axis] + plus*b[w]*a[v] - minus*b[v]*a[w]
        return readout

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    kernel().export_library(str(directory / 'readout.so'))
    (directory / 'metadata.json').write_text(json.dumps(dict(batch=batch, threads=threads)))


class ForestSweepReadout(_PrecompiledLocalReadout):
    """Use reduced Bloch vectors for each connected component merge."""

    def __init__(self, model, directory):
        plan = forest_plan(model)
        if plan is None:
            raise ValueError('This circuit reuses a connected component and requires full evolution.')
        super().__init__(model, directory, plan)

    def transform(self, values):
        qubits = self.result.shape[1]
        result = super().transform(values).reshape(len(values), qubits, 3)
        return result[:, :, [2, 0, 1]].reshape(len(values), 3*qubits)
