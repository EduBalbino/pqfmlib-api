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

## Fold data in RAM

The notebook shares the two PCA estimators and all six projected roles across circuits with the same PCA width. Each estimator is fitted only on its corresponding training role. The cache is read-only and scoped to the current fold, and the existing fold-preparation future keeps the next fold's classical matrices available. On the first full fold, the 139,281 projected rows at width 50 occupy 55,712,400 bytes, about 53 MiB. This avoids repeated covariance eigendecompositions and projections while preserving every row and the stable PCA orientation.

When multiple GPUs are visible, the CPU coordinator prepares the outer representation during the hybrid grid search. The final XGBoost refit still waits for the winning candidate. Single-GPU execution keeps these stages sequential. `PQFM_SIH_RAM_CACHE=0` and `PQFM_SIH_PREFETCH_REFIT=0` disable the respective changes for comparison. Per-trial timing events show the refit overlap, and the shared cache reports its size once per PCA width.

On the hosted 26-qubit disconnected-ring edge-first case at strength 0.10, all four held-out prediction arrays were bitwise identical before and after the change across 14,289 rows. The refit feature stage overlapped the hybrid search by 2.668 s. The new trial took 14.038 s under the sequential five-GPU allocation. The prior 30.138 s measurement used the six-GPU allocation with concurrent trials, so the difference includes scheduling and resource allocation.

The dense queue can increase `PQFM_SIH_QUANTUM_SHARDS` from the default three to the number of sweep GPUs. Each shard evaluates disjoint rows and reassembles the original order. The hosted handoff waits for the five-GPU sweep to finish and the dense worker to write a completed case, then resumes the dense journal on GPUs 0–4 with five shards. GPU 5 becomes available for kernel experiments. Five-way scaling is a planning bound until a full dense case is timed after the handoff.

## Confirmation on untouched folds

Select three standalone-hybrid finalists and two soft-MoE finalists from the completed 69-case screen on fold 1. Evaluate the fixed shortlist on folds 2–15 of the notebook's three repeats of five-fold stratification. Fold 1 serves for selection and is excluded from confirmation summaries. Every case keeps all 27 XGBoost candidates, and all five cases share one 27-candidate classical search per fold. This gives 70 circuit–fold attempts, 1,890 hybrid and 378 classical candidates. These counts refer to grid candidates. Each hybrid candidate includes its ranking fit, exact SHAP selection, and selected-feature fit.

| Circuit | Selected for | Fold 1 AP on selection metric |
|---|---|---:|
| `q19_full_auto_axes_s20` | Hybrid | 0.338588 |
| `q26_plaquettes_auto_axes_s20` | Hybrid | 0.337549 |
| `q26_full_auto_axes_s10` | Hybrid | 0.336869 |
| `q26_native_auto_balanced_s20` | Soft MoE | 0.328150 |
| `q26_plaquettes_auto_shared_s20` | Soft MoE | 0.327942 |

Use a fresh `PQFM_SIH_TOPOLOGY_DIR` containing a copy of the completed manifest and access to the same precompiled artifact directory. Set `PQFM_SIH_TOPOLOGY_COMPLETE=1`, `PQFM_SIH_TOPOLOGY_RUN=1`, and `PQFM_SIH_BACKEND_REVISION=precompiled-v1`. Set `PQFM_SIH_TOPOLOGY_MATCH` to the five table keys separated by commas and `PQFM_SIH_STRATIFIED_FOLDS=2,3,4,5,6,7,8,9,10,11,12,13,14,15`. The hosted confirmation uses GPU 5 and CPU IDs 82–87 while the dense queue continues on GPUs 0–4. Resume with the same journal if interrupted.

Use `python scripts/sih_confirmation_report.py <scores.csv>` to compare AP on matched folds. The report gives paired gaps and win counts against the shared classical baseline, direct comparisons within each objective, and the effect of routing on each hybrid. Repeated folds share admissions across repeats. Interpret their estimates as paired comparisons on this dataset.

## Exhaustive evaluation across all folds

The exhaustive run supersedes the finalist-only execution. It crosses all 90 profiles with all 15 full shuffled folds, for 1,350 circuit–fold attempts and 36,450 hybrid grid candidates. The 69 fast profiles yield 1,035 attempts and the 21 dense profiles yield 315. Each lane shares one classical grid per fold, so running both independently adds at most 810 classical grid candidates across the experiment. Previously completed cases retain their original predictions and timing provenance.

Run the notebook with `PQFM_SIH_TOPOLOGY_COMPLETE=1`, `PQFM_SIH_TOPOLOGY_RUN=1`, `PQFM_SIH_STRATIFIED_FOLDS=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`, and the complete manifest. Leave the topology match unrestricted. The fast lane uses GPU 5 and a separate journal seeded from the 69-case first-fold screen and the completed confirmation attempts. The dense lane resumes its original journal on GPUs 0–4 at the next case boundary. Each lane keeps the same dataset, split roles, and full 27-candidate hybrid grid.

| Measurement | Cases | Median wall per case | Range |
|---|---:|---:|---:|
| Fast sequential, five GPUs | 66 | 14.83 s | 13.39–19.57 s |
| Confirmation, one GPU | 3 | 47.38 s | 46.43–62.80 s |
| Dense, completed full first-fold cases | 4 | 35.04 min | 28.66–36.91 min |

The one-GPU measurements imply roughly 13–17 hours for the remaining fast cases, with uncertainty from the mix of topologies and fold preparation. The dense pilot estimated about 133 GPU-hours per fold from five warm rows per geometry. Across 15 folds that is roughly 2,000 GPU-hours. Dividing by five gives about 17 days under ideal scaling at the measured single-GPU rates. The initial full dense runs do not establish that scaling. The 26-qubit full shared-feature circuit dominates the pilot projection.

## Dense backend hillclimb

The full shared-feature circuit repeats interactions around a small set of high-degree wires. The circuit is a product of edge operators with local preparation. Two edge operators on disjoint wires commute. A schedule that preserves each wire's original edge sequence therefore preserves the unitary. The planner checks that condition for the full-graph schedule and uses the same per-wire dependencies for a bounded search on native layouts. The 19-qubit and 26-qubit full graphs have schedules with four and five simultaneously live wires respectively. Both 26-qubit native layouts also have five-wire schedules.

Let the current density matrix be \(\rho_A\) on the live wires \(A\). A first-use wire enters in its product-state density matrix \(\rho_q\), so the enlarged state is \(\rho_A\otimes\rho_q\). An edge group applies \(U\rho U^\dagger\). Once a wire has no remaining gates, its local observables follow from its reduced density matrix and subsequent gates act only on the remaining wires. Tracing it out at that point leaves all remaining observables unchanged. The TileLang kernel implements these three operations in FP64 shared memory and replaces each retired slot with \(|0\rangle\langle0|\) for reuse. The widest supported frontier uses a \(32\times32\) density matrix per row instead of 67,108,864 amplitudes for the 26-qubit full statevector.

For the three geometries outside the five-wire frontier limit, the revised statevector path batches 32 input rows, builds the fused gate matrices on GPU, and updates the two invariant parity pairs in place for products of \(XX\), \(YY\), and \(ZZ\) rotations. Every amplitude pair is owned by one thread. Products of those rotations preserve parity and have zero entries between the two pairs. Other gates use the original grouped evolution. The offline loader also resolves matrix-builder kernels and pads only the final batch, dropping dummy outputs before returning.

| Geometry | Previous exact readout | Best checked readout | Local speedup |
|---|---:|---:|---:|
| 19-qubit chain shared | 84.83 rows/s | 1,109 rows/s, batch 32 | 13.1× |
| 19-qubit full shared | 59.82 rows/s | 114,902 rows/s, frontier batch 512 / 64 threads | 1,921× |
| 26-qubit full shared | 0.866 rows/s | 36,796 rows/s, frontier batch 256 / 128 threads | 42,476× |

These are warmed RTX 2060 readout measurements, including transfers, with three repeats. The frontier tests processed 4,096 rows per repeat. The statevector tests processed 64 rows for 19 qubits and eight rows for 26 qubits. The two full-circuit frontier checks agreed with the old FP64 TileLang statevector to maximum absolute errors of \(1.9\times 10^{-15}\) and \(2.6\times 10^{-15}\). The speedups apply to readout. Fold training and routing timings are separate.

Build and validate the hosted revision with `python scripts/sih_dense_hill_build.py <manifest.json> <dense-aot-pilot> <output>`. The script checks each geometry against the previous precompiled TileLang reader before emitting `settings.json`. Set `PQFM_SIH_DENSE_TUNING` to this file and set a new backend revision when resuming the dense journal. The hosted handoff pauses the fast lane at a saved case while it uses GPU 5 for the build and readout comparison. It resumes that lane before waiting for a dense checkpoint.

All seven geometry checks passed on one hosted RTX 3070. The builder timed 256 rows per statevector repeat and 4,096 per frontier repeat after warm-up.

| Geometry | Revised hosted rows/s | Readout seconds for 139,281 rows on one GPU |
|---|---:|---:|
| 19-qubit chain | 10,422 | 13.36 |
| 19-qubit full | 230,704 | 0.60 |
| 19-qubit native automatic | 550 | 253.27 |
| 19-qubit native fixed | 260 | 535.79 |
| 26-qubit full | 97,480 | 1.43 |
| 26-qubit native automatic | 109,810 | 1.27 |
| 26-qubit native fixed | 109,022 | 1.28 |

The maximum absolute error versus the hosted batch-one FP64 reader was \(2.7\times10^{-15}\) over the parity samples. These rates project to about 10 GPU-hours of readout for all 315 dense cases, before fold preparation, XGBoost, and scheduling. The projection extrapolates synthetic rows. The five-GPU end-to-end runtime remains to be measured on the resumed sweep. The two 19-qubit native layouts account for about 98% of the revised projected readout time.

The resumed hosted sweep completed the 19-qubit full shared-feature case at strength 0.20 in 20.039 s end to end with all 27 XGBoost candidates. The same geometry at strengths 0.05 and 0.10 took 2,012.209 s and 2,025.526 s with the old backend. These are different strengths and establish a geometry-level timing comparison. The 20.039 s case includes the shared classical branch after resume. The revised dense median requires more cases, particularly the two slower 19-qubit native layouts.

## Paper-aligned strength extension

Set `PQFM_SIH_PAPER_EXTENSION=1` with the complete 90-profile manifest and leave the topology filter unrestricted. The notebook adds strengths 0.025, 0.075, 0.15, and 0.40 to every geometry and encoding. The three previous strengths (0.05, 0.10, and 0.20) retain their original IDs and journal records. The new points probe weaker interactions, fill two intervals, and test a stronger regime. The only circuit parameter that changes is `alpha = strength × blocks`. PCA width 50, repetitions 1, SHAP top 80, the graph, and precompiled kernel signatures are fixed.

| Lane | Profiles | Full folds | Circuit–fold cases | Hybrid grid candidates |
|---|---:|---:|---:|---:|
| Fast | 161 | 15 | 2,415 | 65,205 |
| Dense | 49 | 15 | 735 | 19,845 |
| Total | 210 | 15 | 3,150 | 85,050 |

The extension adds 1,800 cases. It queues the new strengths ahead of unfinished original strengths within each fold. Replay uses the same two journals and exact precompiled TileLang backend, with a separate tuning index for the dense path. Native graph files and compiled libraries must be available at their configured paths. The hosted CUDA libraries target the hosted GPUs. Rebuild the artifacts offline when moving to a different CUDA architecture.

### Comparisons motivated by the fraud-detection paper

The paper trains the experts on balanced data and evaluates them on stratified holdouts at the observed prevalence. It trains a hard XGBoost gate on the indicator that the quantum expert is right and the classical expert is wrong. The SIH-SUS notebook uses this gate target, separates model selection from calibration, router analysis and testing, and compares the gate with soft routing and the standalone experts. The paper's secondary expert is a trained autoencoder/VQC pipeline. The SIH-SUS secondary expert here is XGBoost over classical features and exact Heisenberg observables.

For each newly computed row, the notebook saves raw router probabilities and exports AP, trapezoidal AUPRC, and routing fraction for hard thresholds 0.50 through 0.90 in increments of 0.10. It also saves AP and AUPRC for fixed hybrid weights 0.05, 0.10, 0.25, and 0.50. These are descriptive holdout curves. Select any deployment threshold on training-side validation with a predeclared coverage or precision target. The coverage fraction estimates the potential quantum workload if inference defers circuit readout until after the routing decision. The current experiment computes both experts on all evaluation rows. Previously checkpointed rows retain the diagnostics available at their original run.

The next matched control should use the same classical inputs, training rows, SHAP cap, XGBoost grid and routing protocol, but replace quantum observables with the PCA inputs or a dimension-matched classical feature map. Compare the fixed blend with learned routing to test whether the gate adds value beyond averaging, then compare against the paper-style VQC expert and a matched classical head. Report paired AP gaps on the same folds, precision and recall at an analysis-selected operating point, and AP versus routing coverage. A separate time-ordered validation is needed to estimate performance on future months.
