# SIH-SUS complete topology sweep

The first shuffled, stratified fold keeps the same 14,289 held-out admissions, 50 PCA inputs, SHAP top 80, and \(R=1\). The five circuit layouts per qubit count are chain, disconnected rings (`plaquettes`), full, native automatic, and native fixed. Each uses edge-first independent axes, axis-first independent axes, and a shared feature per edge. Sweep per-block strength \(s\in\{0.05,0.10,0.20\}\) independently for each encoding. The map receives \(\alpha=sB\) for \(B\) encoding blocks so the angle scale \(\alpha/(2B)\) stays fixed across layouts at each strength.

The Cartesian product gives 90 cases per fold and 2,430 hybrid ranking-and-selection candidates. The shared classical grid adds 27 candidates per fold. Each hybrid candidate includes the ranking fit, exact TreeSHAP, and selected-feature fit. The 69 supported cases represent 1,863 hybrid candidates. The remaining 21 cases are three strengths for each of seven shared-feature circuits with exact TileLang cones between 12 and 26 qubits wide. The notebook lists those pending cases explicitly and runs the other 69 with precompiled kernels.

## Running and resuming

Execute the code cells in `sih_sus_stratified_moe_sweep.ipynb` in order. Set `PQFM_SIH_TOPOLOGY_COMPLETE=1`, `PQFM_SIH_STRATIFIED_FOLDS=1`, and `PQFM_SIH_IN_FLIGHT=1`. Use `PQFM_SIH_TOPOLOGY_BUILD=1` for the offline build and `PQFM_SIH_TOPOLOGY_RUN=1` for evaluation. Set `PQFM_SIH_TOPOLOGY_DIR` to a dedicated persistent directory. The hosted run uses CPU affinity 22–81 and physical GPUs 0–4 in PCI order. CPU IDs 82–87 and GPU 5 remain available for backend experiments.

To change TileLang implementations, stop the existing sweep process, retain the output directory, set `PQFM_SIH_BACKEND_REVISION` to a new identifier, build into its revision-specific artifact directory, and restart evaluation. The journal skips completed circuit-and-fold pairs after checking the dataset, protocol, effective model configuration, role assignments, classical device, and XGBoost grid. Bump the protocol identifier if fitting, model selection, or evaluation semantics change. Completed rows record their backend revision. The journal synchronizes each completed case to disk, excludes concurrent writers, and repairs a torn final record on restart. An interrupted case reruns its 27-candidate grid. Segment timing analysis by backend revision and resource allocation. A backend swap assumes numerical equivalence and should be verified against the previous exact implementation on identical inputs.

## Timing on the hosted machine

The previous three hosted batches completed 43 cases in a combined 483.84 s of sweep wall time with three cases in flight and six RTX 3070 GPUs. The resulting 11.25 s throughput per case suggests about 13 min for 69 cases at that allocation, plus setup. The requested sequential allocation has a different throughput. Prior per-case results around 29–43 s imply roughly 33–49 min for 69 sequential cases as an initial planning range. A 10-fold sweep at that rate would take about 5.6–8.2 h for the 69 supported profiles. The first live sample under the requested allocation was faster. Five consecutive cases took 14.92–19.57 s, with a median of 15.07 s. At that median, 69 cases would take about 17.3 min, or 2.9 h for ten folds, before setup. Those five cases all use the 19-qubit chain. The 21 dense cases require separate measurements.

Hosted backend medians across the 43 cases are below. The readout includes host/device copies. GPU-shard times are measured within the transform call, so they exclude queue wait and runner creation. The model-stage rate includes CPU projection, runner creation and queue time. Rates use rows presented to the quantum block, including repeated representations of admissions across fitting roles.

| Backend | Cases | Rows/s divided by sum of shard durations | Rows/s divided by union of shard intervals | Rows/s including projection and scheduling |
|---|---:|---:|---:|---:|
| Causal-cone AOT | 19 | 452,537 | 710,767 | 57,283 |
| Commuting XX | 6 | 963,620 | 1,692,445 | 68,081 |
| Forest | 18 | 731,972 | 1,410,657 | 57,235 |

The winning 26-qubit full edge-first case took 29.150 s. Its 139,281 feature rows took 0.464 s summed across shards, 0.222 s on the readout interval union, and 0.959 s including projection and scheduling. These give about 300,000 rows/s, 627,000 aggregate readout rows/s, and 145,000 model-stage rows/s respectively. The hybrid search took 22.532 s of wall time. Candidate work comprised 20.659 s summed over ranking fits, 8.959 s over exact TreeSHAP, 0.498 s over selected-column binning, and 13.111 s over selected-column fits. These candidate durations overlap across GPUs. The concurrent shared classical branch took 27.556 s for the fold in that batch.

| Feature role | Rows | Projection (s) | GPU readout interval union (s) | Model-stage wall (s) |
|---|---:|---:|---:|---:|
| Inner training | 16,178 | 0.068 | 0.067 | 0.232 |
| Validation | 45,725 | 0.190 | 0.046 | 0.248 |
| Expert refit | 20,222 | 0.084 | 0.026 | 0.130 |
| Calibration | 28,578 | 0.121 | 0.031 | 0.171 |
| Router analysis | 14,289 | 0.058 | 0.029 | 0.094 |
| Held-out test | 14,289 | 0.059 | 0.023 | 0.084 |

Other wall stages on this case included 0.436 s for the inner PCA fit, 0.511 s for the refit PCA, 0.640 s for inner hybrid scaling, 0.535 s for refit scaling, 0.580 s for the outer ranking fit, 0.280 s for outer SHAP, 0.306 s for the outer prediction fit, and 1.237 s for the router. The 29.150 s is total trial time and these measurements overlap with the shared classical branch.

## Dense queue

Build the dense readout libraries with `python scripts/sih_dense_aot_pilot.py <topology-dir>` on the target GPU. Pass the resulting `dense-aot-pilot` directory in `PQFM_SIH_DENSE_AOT`, set `PQFM_SIH_BACKEND_REVISION`, and run the same notebook. This selects the previously pending cases and writes a separate `dense-results-tilelang-rows0.jsonl` journal so it can run next to the five-GPU sweep. The hosted dense worker uses GPU 5 and CPU IDs 82–87. The evaluator loads ahead-of-time libraries and reports a missing kernel as an error. The pilot runs at batch size one with CPU gate-matrix preparation. It times five warm rows per geometry.

| Circuit | Rows/s | Estimated feature hours per strength |
|---|---:|---:|
| 19-qubit chain | 74.60 | 0.52 |
| 19-qubit full | 47.09 | 0.82 |
| 19-qubit native automatic | 69.19 | 0.56 |
| 19-qubit native fixed | 60.43 | 0.64 |
| 26-qubit full | 1.36 | 28.35 |
| 26-qubit native automatic | 14.28 | 2.71 |
| 26-qubit native fixed | 3.59 | 10.78 |

These are throughput extrapolations from five synthetic feature rows per circuit. The 21 queued cases would require roughly 133 GPU-hours of readout at these rates. The 26-qubit full geometry accounts for roughly 85 of those hours. The pilot establishes execution feasibility. Held-out quality requires completed fold runs.
