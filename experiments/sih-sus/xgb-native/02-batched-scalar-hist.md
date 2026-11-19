# Batched scalar child histograms

The patch targets upstream XGBoost v3.2.0 at `7991260512353c8f408adf2e2cea1a0d0b4b4468`. It changes scalar histogram construction for in-memory, single-process trees when one expansion batch supplies more than one child histogram. The root, singleton batches, overflow rebuild branch, distributed and external-memory paths retain stock dispatch.

The original kernel builds one child histogram per launch. A worker now builds a prefix sum of each selected child's block count and launches a grid for up to 32 children at a time. Feature groups retain their existing grid-y mapping. Each block finds its child by the prefix sum, computes the same child-local tile offsets and stride, and invokes the existing `HistKernelOneNodeTarget` with that child's row and histogram spans. The kernel retains scalar launch bounds, shared versus global selection, bin access, and integer gradient accumulation. Child block count can change if the new kernel's register use affects the occupancy cap. No temporary device metadata buffer is allocated because descriptors are passed in a sub-2 KiB grid-constant kernel argument. The host still waits for partition row counts before constructing the grid.

A child histogram is the sum of its fixed-point pairs over the same set of rows in each bin. Each feature group and child uses disjoint output regions, and blocks within a child can update the same accumulator with integer atomic additions. The block map covers each grid-stride tile once, so changing the order of the integer updates preserves the histogram. Grid-prefix lookup and the larger per-launch payload are potential overheads. Measure the actual CUDA child-histogram launch counts, shared-memory and register occupancy, and fit runtime by tree depth. Measure the whole candidate, including unchanged SHAP and AP, to distinguish a fit gain from a search gain.

## Reproduce

The patch applies on the exact upstream release. This builds the source on a CUDA 12.9 host targeting V100. Use SM75 on a local CUDA 13 RTX 2060 system. Build one stock source baseline with the same compiler configuration if comparing timing.

```sh
git apply --check /path/to/02-batched-scalar-hist.patch
git apply /path/to/02-batched-scalar-hist.patch
cmake -S . -B build -GNinja -DUSE_CUDA=ON -DUSE_NCCL=OFF -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build
```

For the standalone segmented-grid check on RTX 2060:

```sh
nvcc -std=c++17 -arch=sm_75 -O2 /path/to/02-segmented-grid-check.cu -o /tmp/segmented-grid-check
/tmp/segmented-grid-check
```

For same-device behavioral parity, run the harness in environments that load stock and patched XGBoost v3.2.0, respectively. Each run fits a ranking model, computes exact TreeSHAP on all train rows, selects the ordered top 80, fits a second model, and computes validation AP. It compares both serialized model hashes, ordered masks, prediction byte hashes, and AP on dense and missing-valued synthetic inputs.

```sh
PYTHONPATH=/path/to/stock/python-package python /path/to/02-batched-hist-parity.py /tmp/stock.json
PYTHONPATH=/path/to/patched/python-package python /path/to/02-batched-hist-parity.py /tmp/patched.json --reference /tmp/stock.json
```

## Verified

- Patch apply check and whitespace check passed on the tagged source.
- Both edited CUDA translation units compiled from v3.2.0 with CUDA 13.0 targeting SM75.
- Standalone SM75 CUDA ran 71 node histograms including nine empty nodes, checking 13,632 integer pairs against CPU sums. It issued two GPU kernels for the 62 nonempty nodes. This check covers segmented mapping, capped block counts, empty nodes, negative fixed-point gradients, and missing bins in the standalone model.
- The end-to-end parity harness completed both synthetic cases on the stock 3.2.0 wheel.
- Full linked patched-library parity and V100 timings are pending. The standalone CUDA check uses a synthetic histogram kernel, while the patch calls XGBoost's existing accumulator and dispatches the result through the actual booster.
