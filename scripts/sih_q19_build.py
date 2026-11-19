"""Build and compare the three 19-qubit dense readouts on the target GPU."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from pqfmlib import HeisenbergProjectiveQFM
from pqfmlib.experimental.heisenberg_sih_aot import SIHHeisenbergAOT, build_sih_kernels, plan_sih_circuit
from pqfmlib.experimental.sih_frontier_tilelang import FrontierSweepReadout, frontier_plan
from pqfmlib.experimental.sih_topology_sweep import GenericTileLangSweepReadout
from pqfmlib.experimental.sih_wide_frontier_tilelang import build_wide_frontier, wide_frontier_shape


def build_q19_revision(manifest_path, previous_settings, output):
    manifest = json.loads(manifest_path.read_text())
    settings = json.loads(previous_settings.read_text())
    values = np.random.default_rng(2718).normal(size=(521, 50))
    for geometry in ('q19_native_auto_shared', 'q19_native_fixed_shared', 'q19_chain_auto_shared'):
        target = output / geometry
        current = settings[geometry]
        candidates = []
        for suffix in ('s025', 's10', 's400'):
            template = manifest[geometry + '_s10']
            strength = {'s025': .025, 's10': .10, 's400': .40}[suffix]
            config = {k: v for k, v in template['config'].items() if k != 'n_components'}
            config['alpha'] = strength * template['blocks']
            model = HeisenbergProjectiveQFM(
                name_file='dense-q19', **config, R=1, seed=42, framework='tilelang',
                expectation_method='exact', save_circuit_drawings=False,
            ).fit(values[:1])
            baseline = GenericTileLangSweepReadout(
                model, artifacts=current['path'], **current['options'])
            if '_chain_' in geometry:
                if not (target / 'metadata.json').is_file():
                    shape, *_ = plan_sih_circuit(model)
                    build_sih_kernels(target, **shape, batch=256, compact=True)
                candidate = SIHHeisenbergAOT(model, target)
                mode = dict(cones=True)
            else:
                if not (target / 'metadata.json').is_file():
                    plan = frontier_plan(model, max_width=7, compact=True)
                    build_wide_frontier(target, *wide_frontier_shape(model, plan), batch=256, threads=256, compact=True)
                candidate = FrontierSweepReadout(model, target)
                mode = dict(frontier=True, width=7)
            expected = baseline.transform(values[:17])
            result = candidate.transform(values[:17])
            np.testing.assert_allclose(result, expected, atol=1e-10, rtol=0)
            error = float(np.max(np.abs(result - expected)))
            if suffix != 's10':
                candidates.append(dict(strength=strength, max_abs_error=error))
                continue
            old_times, new_times = [], []
            baseline.transform(values)
            candidate.transform(values)
            for _ in range(3):
                for readout, timings in ((baseline, old_times), (candidate, new_times)):
                    start = time.perf_counter()
                    readout.transform(values)
                    timings.append(time.perf_counter() - start)
            speeds = dict(old_rows_per_s=len(values) / statistics.median(old_times),
                          rows_per_s=len(values) / statistics.median(new_times))
            candidates.append(dict(strength=strength, max_abs_error=error, **speeds))
        record = dict(**mode, **speeds, max_abs_error=max(row['max_abs_error'] for row in candidates),
                      path=str(target.resolve()), parity=candidates, benchmark_rows=len(values))
        adopt = speeds['rows_per_s'] > speeds['old_rows_per_s'] * 1.05
        if adopt:
            settings[geometry] = record
        print('Q19_HILL_RESULT', geometry, json.dumps(record | dict(adopt=adopt)), flush=True)
    (output / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('previous_settings', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build_q19_revision(args.manifest, args.previous_settings, args.output)
