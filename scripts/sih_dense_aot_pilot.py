"""Build exact dense TileLang libraries and time one profile per geometry."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tilelang import tvm

from pqfmlib import HeisenbergProjectiveQFM
from pqfmlib.experimental.sih_topology_sweep import GenericTileLangSweepReadout


def build_and_measure(root):
    target = root / 'dense-aot-pilot'
    target.mkdir(exist_ok=True)
    cache = {}
    rows = np.random.default_rng(42).normal(size=(5, 50))
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, entry in manifest.items():
        if entry['backend'] != 'generic_tilelang' or not name.endswith('_s10'):
            continue
        try:
            config = {key: value for key, value in entry['config'].items() if key != 'n_components'}
            model = HeisenbergProjectiveQFM(
                name_file='dense-pilot', **config, framework='tilelang',
                expectation_method='exact', R=1, seed=42, save_circuit_drawings=False,
            ).fit(rows[:1])
            readout = GenericTileLangSweepReadout(model, profile=True, batch_size=1)
            executor = readout.executor
            executor.matrix_backend = 'cpu'

            def launch(kernel, *signature):
                key = (kernel, *signature)
                if key not in cache:
                    library = target / ('_'.join(map(str, key)) + '.so')
                    if not library.exists():
                        getattr(executor._kernels, kernel)(*signature).export_library(str(library))
                    cache[key] = tvm.runtime.load_module(str(library))
                return cache[key]

            executor._launcher = launch
            start = time.perf_counter()
            readout.transform(rows[:1])
            build = time.perf_counter() - start
            start = time.perf_counter()
            result = readout.transform(rows)
            seconds = time.perf_counter() - start
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite dense readout')
            rate = len(rows) / seconds
            record = dict(build_and_warm_s=build, rows=len(rows), seconds=seconds,
                          rows_per_s=rate, feature_stage_hours=139281 / rate / 3600,
                          stats=executor.last_execution_stats)
            print('DENSE_PILOT', name, json.dumps(record), flush=True)
        except (RuntimeError, OSError, ValueError) as error:
            print('DENSE_PILOT_FAILED', name, repr(error), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Directory containing the complete topology manifest')
    build_and_measure(parser.parse_args().root.resolve())
