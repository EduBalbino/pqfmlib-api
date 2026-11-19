"""Exact fixed-point GPU histogram prototype using cuts from real fold rows."""
import json
import os
from pathlib import Path
from statistics import median
import cupy as cp
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

source = Path(os.environ.get('PQFM_SIH_BENCH_ROOT', 'outputs/sih-v100')) / 'xgb/fold.npz'
with np.load(source) as fold:
    x = np.column_stack((fold['projected_train'], fold['train']))
    y = fold['y']
x = StandardScaler().fit_transform(x)
assert np.isfinite(x).all()
indptr, cuts = xgb.QuantileDMatrix(x, y, nthread=2).get_quantile_cut()
indptr, cuts = indptr.astype(np.int32), cuts.astype(np.float32)
codes = np.column_stack([
    np.minimum(np.searchsorted(cuts[a:b], col, side='right'), b-a-1)+a
    for col, a, b in zip(x.T, indptr[:-1], indptr[1:])
]).astype(np.int32)
nrows, ncols = codes.shape
nbins = int(indptr[-1])
modes = np.array([
    np.bincount(codes[:, j]-a, minlength=b-a).argmax()+a
    for j, (a,b) in enumerate(zip(indptr[:-1],indptr[1:]))
], dtype=np.int32)
# Keep all bins for columns where the mode does not dominate.
dominant = np.count_nonzero(codes == modes, axis=0) >= .90*nrows
modes = np.where(dominant, modes, -1).astype(np.int32)
rows, cols = np.nonzero(codes != modes)
exceptions = codes[rows, cols]
rng = np.random.default_rng(42)
gradients = rng.integers(-2**30, 2**30, size=(nrows,2),dtype=np.int64)
g_codes, g_modes, g_indptr, g_grad, g_rows, g_ex = map(cp.asarray,
    (codes,modes,indptr,gradients,rows.astype(np.int32),exceptions))
module = cp.RawModule(code=r'''
typedef unsigned long long U64;
__device__ void push(U64* dest, const U64* src) {
    atomicAdd(dest, src[0]); atomicAdd(dest+1, src[1]);
}
extern "C" __global__ void dense(const int* codes,const U64* gp,const int* node,
                                U64* h,int count,int cols,int bins) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<count) {
        int r=i/cols;
        push(h+2*(node[r]*bins+codes[i]),gp+2*r);
    }
}
extern "C" __global__ void excepts(const int* rows,const int* bins_idx,const U64* gp,
                                  const int* node,U64* h,int count,int bins) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<count) {
        int r=rows[i];
        push(h+2*(node[r]*bins+bins_idx[i]),gp+2*r);
    }
}
extern "C" __global__ void totals(const U64* gp,const int* node,U64* h,int count) {
    int r=blockDim.x*blockIdx.x+threadIdx.x;
    if(r<count) push(h+2*node[r],gp+2*r);
}
extern "C" __global__ void restore(const int* modes,const int* ptr,const U64* totals,
                                  U64* h,int n,int cols,int bins) {
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i>=n*cols) return;
    int node=i/cols,j=i%cols,m=modes[j];
    if(m<0) return;
    U64 g=totals[2*node],s=totals[2*node+1];
    U64* dst=h+2*node*bins;
    for(int b=ptr[j];b<ptr[j+1];++b) if(b!=m) {g-=dst[2*b];s-=dst[2*b+1];}
    dst[2*m]=g;dst[2*m+1]=s;
}
''')
def launch(name, size, *args):
    module.get_function(name)(((size+255)//256,), (256,), args)
def measure(fn, repeats=7):
    samples=[]
    for _ in range(repeats):
        start, end = cp.cuda.Event(),cp.cuda.Event()
        start.record(); fn(); end.record(); end.synchronize()
        samples.append(cp.cuda.get_elapsed_time(start,end))
    return median(samples)
def i32(*values):
    return tuple(map(np.int32,values))
print(json.dumps(dict(rows=nrows, columns=ncols, bins=nbins, dominant_columns=int(dominant.sum()),
                      dense_entries=codes.size, exceptions=len(rows),
                      zeros_after_scaling=int(np.count_nonzero(x==0)))),flush=True)
for n in (1,16,128):
    g_node=cp.asarray(rng.integers(0,n,size=nrows,dtype=np.int32))
    a,b=cp.zeros((2,n,nbins,2),dtype=cp.int64)
    t=cp.zeros((n,2),dtype=cp.int64)
    def full():
        a.fill(0)
        launch('dense',codes.size,g_codes,g_grad,g_node,a,*i32(codes.size,ncols,nbins))
    def optimized():
        b.fill(0); t.fill(0)
        launch('totals',nrows,g_grad,g_node,t,*i32(nrows))
        launch('excepts',len(rows),g_rows,g_ex,g_grad,g_node,b,*i32(len(rows),nbins))
        launch('restore',n*ncols,g_modes,g_indptr,t,b,*i32(n,ncols,nbins))
    full(); optimized()
    cp.testing.assert_array_equal(a,b)
    dense_ms,opt_ms=measure(full),measure(optimized)
    print(json.dumps(dict(nodes=n,dense_ms=dense_ms,dominant_ms=opt_ms,
                          prototype_speedup=dense_ms/opt_ms,bit_exact=True)),flush=True)
