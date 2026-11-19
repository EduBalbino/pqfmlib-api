"""Run from the hosted SIH directory with the existing xgb-hill environment."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import os
from pathlib import Path
from statistics import median
import time

bench = Path(os.environ.get('PQFM_SIH_BENCH_ROOT', 'outputs/sih-v100')).resolve() / 'xgb-hill/bench.py'
namespace = {'__file__': str(bench)}
exec(compile(bench.read_text().split('\nmatrices=')[0], str(bench), 'exec'), namespace)
cp, np, xgb, X, V, Y, vy, grid, ap = (
    namespace[key] for key in ('cp', 'np', 'xgb', 'X', 'V', 'Y', 'vy', 'grid', 'average_precision_score')
)

def batch(flag):
    from threading import local
    state = local()
    def candidate(index):
        with cp.cuda.Device(0), xgb.config_context(use_cuda_async_pool=flag):
            if not hasattr(state, 'inputs'):
                state.inputs = (
                    xgb.QuantileDMatrix(X, Y, nthread=2),
                    [xgb.DMatrix(X[i:i+8192], nthread=2) for i in range(0, len(X), 8192)],
                )
            base, shap_data = state.inputs
            eta, depth, weight = grid[index]
            params = dict(device='cuda:0', objective='binary:logistic', tree_method='hist',
                          max_depth=depth, learning_rate=eta, min_child_weight=weight,
                          subsample=.85, colsample_bytree=.85, nthread=2, seed=42)
            ranking = xgb.train(params, base, num_boost_round=300)
            totals = sum(np.abs(ranking.predict(mat, pred_contribs=True)[:,:-1]).sum(axis=0, dtype=np.float64)
                         for mat in shap_data)
            mask = np.argsort(-totals, kind='stable')[:80]
            selected = xgb.QuantileDMatrix(X[:, mask], Y, nthread=2)
            model = xgb.train(params, selected, num_boost_round=300)
            prediction = model.inplace_predict(V[:, mask]).get()
            return sha256(ranking.save_raw()).hexdigest(), sha256(model.save_raw()).hexdigest(), mask.tolist(), ap(vy,prediction)
    start = time.perf_counter()
    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(candidate, (0,4,8,18,22,26)))
    return time.perf_counter()-start, results

print('START', xgb.__version__, X.shape, V.shape, flush=True)
reference = None
samples = {False: [], True: []}
for enabled in (False, True, True, False):
    duration, outcome = batch(enabled)
    if reference is None:
        reference = outcome
    equal = outcome == reference
    print('BATCH', enabled, round(duration, 4), 'same_models_masks_ap', equal, flush=True)
    samples[enabled].append(duration)
    if not equal:
        print('PARITY_DIFF', [(i, a[0]==b[0], a[1]==b[1], a[2]==b[2], float(a[3]-b[3]))
                              for i,(a,b) in enumerate(zip(reference,outcome)) if a!=b],flush=True)
print('DONE', {str(flag):median(vals) for flag,vals in samples.items()}, flush=True)
