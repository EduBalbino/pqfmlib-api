# SIH-SUS Heisenberg experiments

Open `sih_sus_reference_moe_sweep.ipynb` in JupyterLab to inspect or run the experiment. The notebook uses the frozen SIH-SUS admissions data and protocol under `data/sih-sus/protocolo_comparacao`, all available training rows, ten monthly test folds, and a TileLang Heisenberg expert. It fits a soft router using historical Analysis months and saves predictions and fitted models under `outputs/reference_soft_moe_sweep`.

The 2025 test months were previously inspected, so results are exploratory. The SIH-SUS test folds are one month each; the SIA-SUS PDF used six-month test folds. This notebook does not implement the PDF's separate no-Analysis baseline.

The sweep implementation is contained in the notebook and uses the shared `pqfmlib` TileLang backend.

## Shuffled stratified sweep

Open `sih_sus_stratified_moe_sweep.ipynb` to compare models on randomly held-out admissions. It uses three repeats of stratified five-fold splitting. Expert fitting uses minority/majority balance, and calibration, router analysis, and final test preserve the observed class proportions. The temporal and shuffled protocols answer different questions. Compare AP within each protocol.

The notebook reports classical XGBoost, Heisenberg features plus classical features in a single XGBoost expert, and hard and soft routing. The standalone expert's score uses its own test predictions. Router analysis is used for the MoE scores.

The chain connects consecutive qubits end to end. The disconnected-ring layout closes groups of up to four qubits into cycles, with a single edge for a terminal pair. Gates stay within each group. The 19-qubit ring layout has four squares and a triangle. The 26-qubit layout has six squares and one pair. Both layouts use balanced axis encoding with 50 PCA inputs, one block, one circuit pass, and `alpha=0.1`. In the 26-qubit comparison, 25 edges carry one `XX` and one `YY` component each. At 19 qubits, both layouts also populate some `ZZ` slots to fit the same inputs.

For the first-fold 19-qubit backfill, set `PQFM_SIH_STRATIFIED_ENCODINGS=q19_chain,q19_plaquettes` and `PQFM_SIH_STRATIFIED_FOLDS=1` in the kernel environment. Run the notebook with `PQFM_SIH_BUILD_STRATIFIED=1` to compile the two GPU artifacts offline. For the measured run, switch off the build flag and set `PQFM_SIH_RUN_STRATIFIED=1`. Use `PQFM_SIH_STRATIFIED_AOT` for the artifact directory and `PQFM_SIH_STRATIFIED_RESULTS` for a separate JSONL journal.

The 19-qubit backfill used the same dataset checksum, first shuffled fold, and 27 XGBoost candidates as the hosted 26/33-qubit comparison. Classical predictions matched exactly on all 14,289 test admissions.

| Circuit | Classical XGBoost AP | Hybrid XGBoost AP | Soft MoE AP |
| --- | ---: | ---: | ---: |
| 19 qubits, chain | 0.326233 | 0.324648 | 0.326414 |
| 19 qubits, disconnected rings | 0.326233 | 0.323553 | 0.327033 |
| 26 qubits, disconnected rings | 0.326233 | 0.331384 | 0.326724 |

The two 19-qubit configurations took 396.26 seconds together on one local RTX 2060, with offline compilation excluded. On the hosted machine, the two 26-qubit configurations took 28.15 seconds together with two trials in flight. Those measurements use different hardware and concurrency settings.
