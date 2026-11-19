"""Bench #5's reduction on real contributions. Stock XGBoost supplies those via host."""
import os
from pathlib import Path
from statistics import median
import json
import time
import cupy as cp
import numpy as np
import xgboost as xgb

with np.load(Path(os.environ.get('PQFM_SIH_BENCH_ROOT', 'outputs/sih-v100')) / 'xgb/fold.npz') as f:
    x = cp.asarray(np.column_stack((f['projected_train'], f['train'])))
    y = cp.asarray(f['y'])
booster = xgb.train(dict(device='cuda:0', tree_method='hist', max_depth=4, nthread=2,
                         subsample=.85, colsample_bytree=.85, objective='binary:logistic'),
                    xgb.QuantileDMatrix(x,y,nthread=2), num_boost_round=300)
blocks = [booster.predict(xgb.DMatrix(x[i:i+8192],nthread=2),pred_contribs=True)
          for i in range(0,len(y),8192)]
gpu = list(map(cp.asarray, blocks))
width = blocks[0].shape[1]-1
kernel = cp.RawKernel(r'''
extern "C" __global__ void reduce(const float* values, double* dst, int rows, int width) {
    int col=blockDim.x*blockIdx.x+threadIdx.x;
    if (col>=width) return;
    double sum=0;
    for (int row=0;row<rows;++row) {
      double v=values[(long long)row*(width+1)+col];
      sum+=v<0?-v:v;
    }
    dst[col]=sum;
}
''', 'reduce')
output = cp.empty(width,dtype=cp.float64)
def reduction():
    totals = np.zeros(width,dtype=np.float64)
    for block in gpu:
        kernel(((width+255)//256,), (256,), (block,output,np.int32(len(block)),np.int32(width)))
        totals += output.get()
    return totals
def exact_host():
    return sum(np.abs(block[:,:-1]).sum(axis=0,dtype=np.float64) for block in blocks)
def stock_return_path():
    return sum(np.abs(block.get()[:,:-1]).sum(axis=0,dtype=np.float64) for block in gpu)
reference = exact_host()
result = reduction()
np.testing.assert_array_equal(reference,result)
np.testing.assert_array_equal(np.argsort(-reference,kind='stable')[:80],np.argsort(-result,kind='stable')[:80])
samples = [[],[]]
for _ in range(7):
    for index, fn in enumerate((stock_return_path,reduction)):
        start=time.perf_counter()
        fn()
        samples[index].append(time.perf_counter()-start)
print(json.dumps(dict(rows=len(y), columns=width, blocks=len(blocks), exact=True,
                      baseline_bytes=sum(b.nbytes for b in blocks), reduced_bytes=len(blocks)*result.nbytes,
                      baseline_return_ms=1000*median(samples[0]), reduced_return_ms=1000*median(samples[1])),indent=2))
