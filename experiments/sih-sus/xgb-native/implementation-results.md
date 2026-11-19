# XGBoost hybrid search implementation pass

These experiments target the ranking and selected-feature fits and exact TreeSHAP. The live sweep kept running on its original configuration during the benchmarks.

## 1. Stock CUDA async pool

The hosted XGBoost 3.2.0 A/B replayed six 300-round hybrid candidates twice per setting on a V100 with two workers. It used real q33 native TileLang readouts and the frozen SIH fold. Median batch time was 18.1785 s without the pool and 13.8021 s with it (1.317× throughput, 24.1% less wall time). The four batches had identical ranking/final booster hashes, ordered top-80 masks, and validation AP. The ongoing sweep shared the device, so this is a short contention-sensitive comparison. The notebook exposes the worker-local mode as `PQFM_XGB_CUDA_ASYNC_POOL=1` and records it in new results.

## 2. Scalar child-histogram batching

The isolated v3.2.0 patch makes one kernel dispatch for up to 32 scalar child-node jobs, retaining the stock scalar kernel body, fixed-point accumulation, and normal root/fallback paths. Its standalone CUDA dispatch check covered 71 nodes, including empty nodes and missing bins, and matched the reference exactly. The patch does not remove the row-partition host fence. See the companion native patch and parity harness for full-library verification.

## 4. Dominant observed-bin reconstruction

The standalone CUDA test used real, centered SIH features with 50 frozen PCA columns as proxy quantum features, local QuantileDMatrix cuts, and generated fixed-point gradients and node assignments. The input was fully nonzero after scaling. Of 1,104 columns, 1,033 had at least 90% occupancy in one bin. The prototype retained 1,212,100 exceptions from 17,860,512 entries and matched every integer histogram pair for 1, 16, and 128 nodes.

The local 1/16/128-node prototype timings were 4.40/4.40/8.02 ms with simple direct CUDA atomics and 0.23/0.60/2.66 ms with dominant-bin reconstruction. These compare kernels inside the standalone prototype, exclude setup, and do not compare with the optimized stock XGBoost histogram implementation. Integration requires wiring this representation into stock ELLPACK histogram construction and testing full training equivalence.

## 5. GPU-side exact SHAP reduction

The tagged CUDA C API translation unit compiled with an experimental entry point that reduces exact contributions in deterministic row order on device and returns FP64 feature totals. The Python ABI wrapper is provided. The standalone test ran exact TreeSHAP on the local SIH proxy and checked the reduction on two genuine output matrices on an RTX 2060. The score vector and ordered top 80 matched bit for bit. The reduced return path was 8.11 ms versus 168.48 ms for the full-copy-plus-CPU-reduction path, with 17,664 bytes transferred versus 71,506,760 bytes. This excludes TreeSHAP inference and is not an end-to-end patched-library timing.

## Reproducing the recorded checks

Run from the repository root with the experiment environment. The Python harnesses use
`outputs/sih-v100` by default. Set `PQFM_SIH_BENCH_ROOT=/workspace/sih-v100` on the hosted
machine so the allocator replay can load `xgb-hill/bench.py`, its recorded fold and the
compiled q33 readout. The dominant-bin and SHAP checks require `xgb/fold.npz`. The sample
outputs are included beside their scripts. The experimental native patches target
XGBoost v3.2.0 at `7991260512353c8f408adf2e2cea1a0d0b4b4468`.

## Hosted sweep restart

On 2026-11-19 at 13:10 UTC, the hosted launcher resumed from 762 saved dense
results using the same journal and case keys, with three trials in flight and
`PQFM_XGB_CUDA_ASYNC_POOL=1`. The new launcher PID was 19589 and the dashboard
publisher PID was 19591. The notebook worker environment confirmed the allocator
flag. The native histogram and SHAP prototypes remain separate from the live sweep.
