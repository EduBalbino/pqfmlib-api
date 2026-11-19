"""Build and parity-check faster exact TileLang kernels for the dense SIH-SUS queue."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from tilelang import tvm

from pqfmlib import HeisenbergProjectiveQFM
from pqfmlib.experimental.sih_frontier_tilelang import (
    FrontierSweepReadout, build_frontier, frontier_plan, frontier_shape,
)
from pqfmlib.experimental.sih_topology_sweep import GenericTileLangSweepReadout


def build_dense_revision(manifest_path, baseline, output):
    output.mkdir(parents=True, exist_ok=True)
    values = np.random.default_rng(42).normal(size=(4096, 50))
    manifest = json.loads(manifest_path.read_text())
    settings = {}
    for name, row in manifest.items():
        if row['backend'] != 'generic_tilelang' or not name.endswith('_s10'):
            continue
        geometry = name.rsplit('_', 1)[0]
        config = {key: value for key, value in row['config'].items() if key != 'n_components'}
        model = HeisenbergProjectiveQFM(
            name_file='dense-hill', **config, R=1, seed=42, framework='tilelang',
            expectation_method='exact', save_circuit_drawings=False,
        ).fit(values[:1])
        truth = GenericTileLangSweepReadout(model, artifacts=baseline).transform(values[:5])
        plan = frontier_plan(model)
        target = output / geometry
        target.mkdir(exist_ok=True)
        if plan is not None:
            shape = frontier_shape(model, plan)
            if not (target / 'metadata.json').exists():
                build_frontier(target, *shape, batch=256, threads=128)
            evaluator = FrontierSweepReadout(model, target)
            mode = dict(frontier=True, width=shape[2])
            probe = values[:4096]
        else:
            options = dict(batch_size=32, matrix_backend='gpu', parity_gates=True)
            builder = GenericTileLangSweepReadout(model, **options)
            executor = builder.executor
            compiler = executor._launcher

            def export(kernel, *signature):
                library = target / ('_'.join(map(str, (kernel, *signature))) + '.so')
                if not library.exists():
                    compiler(kernel, *signature).export_library(str(library))
                return tvm.runtime.load_module(str(library))

            executor._launcher = export
            builder.transform(values[:32])
            evaluator = GenericTileLangSweepReadout(model, artifacts=target, **options)
            mode = dict(frontier=False, options=options)
            probe = values[:256]
        got = evaluator.transform(values[:5])
        np.testing.assert_allclose(got, truth, atol=1e-10, rtol=0)
        times = []
        for _ in range(3):
            started = time.perf_counter()
            evaluator.transform(probe)
            times.append(time.perf_counter() - started)
        settings[geometry] = dict(**mode, rows_per_s=len(probe) / statistics.median(times),
                                  max_abs_error=float(np.max(np.abs(got - truth))),
                                  path=str(target.resolve()))
        print('DENSE_HILL_VALIDATED', geometry, json.dumps(settings[geometry]), flush=True)
    (output / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build_dense_revision(args.manifest, args.baseline, args.output)
