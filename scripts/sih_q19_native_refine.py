"""Compare identical native readouts offline and retain only measured improvements."""
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

from pqfmlib import HeisenbergProjectiveQFM
from pqfmlib.experimental.sih_frontier_tilelang import FrontierSweepReadout, frontier_plan
from pqfmlib.experimental.sih_wide_frontier_tilelang import build_wide_frontier, wide_frontier_shape


def benchmark(manifest_file, previous_file, output):
    manifest = json.loads(manifest_file.read_text())
    previous = json.loads(previous_file.read_text())
    settings = previous.copy()
    rows = np.random.default_rng(49).normal(size=(2051, 50))
    for layout in ('auto', 'fixed'):
        geometry = f'q19_native_{layout}_shared'
        template = manifest[f'{geometry}_s10']
        config = {k: v for k, v in template['config'].items() if k != 'n_components'}

        def fitted(strength):
            return HeisenbergProjectiveQFM(
                name_file='native-refine', **(config | {'alpha': strength * template['blocks']}),
                R=1, seed=42, framework='tilelang', expectation_method='exact',
                save_circuit_drawings=False,
            ).fit(rows[:1])

        model = fitted(.10)
        shape = wide_frontier_shape(model, frontier_plan(model, max_width=7, compact=True))
        candidates = {'v3': Path(previous[geometry]['path'])}
        for exchange in (False, True):
            name = 'tied_exchange' if exchange else 'bit_index'
            path = output / name
            if not (path / 'metadata.json').exists():
                build_wide_frontier(path, *shape[:-1], shape[-2], batch=256, threads=256,
                                    compact=True, exchange=exchange)
            candidates[name] = path
        results = {name: {'times': [], 'max_abs_error': 0.0} for name in candidates}
        for strength in (.025, .10, .40):
            model = fitted(strength)
            readers = {name: FrontierSweepReadout(model, path) for name, path in candidates.items()}
            expected = readers['v3'].transform(rows[:259])
            for name, reader in readers.items():
                actual = reader.transform(rows[:259])
                np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=0)
                results[name]['max_abs_error'] = max(results[name]['max_abs_error'],
                                                    float(np.max(np.abs(actual - expected))))
            if strength == .10:
                for _ in range(5):
                    for name, reader in readers.items():
                        start = time.perf_counter()
                        reader.transform(rows)
                        results[name]['times'].append(time.perf_counter() - start)
        for result in results.values():
            result['rows_per_s'] = len(rows) / statistics.median(result['times'])
        best = max(results, key=lambda name: results[name]['rows_per_s'])
        before, after = results['v3']['rows_per_s'], results[best]['rows_per_s']
        print('NATIVE_REFINED', geometry, best, json.dumps(results), flush=True)
        if best != 'v3' and after >= before * 1.05:
            settings[geometry] = previous[geometry] | dict(path=str(candidates[best].resolve()),
                                                            rows_per_s=after, old_rows_per_s=before,
                                                            native_optimization=best, benchmark_rows=len(rows),
                                                            max_abs_error=results[best]['max_abs_error'])
    (output / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')


if __name__ == '__main__':
    benchmark(*map(Path, sys.argv[1:]))
