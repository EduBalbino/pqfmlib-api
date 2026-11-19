# SIH-SUS hybrid XGBoost grid selection

The audit covers 1,532 distinct hybrid searches with scores for all 27 candidates,
including 910 records from the current dense sweep and 622 archived searches.
Candidate selection uses inner-validation average precision, with the original
candidate order breaking ties. Mirrored copies of the same search are counted
once. The candidate statistics and replay CSVs also include the classical branch.

| Learning rate | Max depth | Min child weight | Dense wins | All wins |
|---:|---:|---:|---:|---:|
| 0.05 | 4 | 5 | 886 | 1,486 |
| 0.05 | 4 | 10 | 17 | 31 |
| 0.05 | 4 | 20 | 0 | 4 |
| 0.05 | 6 | 5 | 5 | 9 |
| 0.10 | 4 | 5 | 1 | 1 |
| 0.10 | 4 | 10 | 1 | 1 |

These six candidates recover every observed winner retrospectively and reduce
candidate fits by 21/27. A fixed 0.05 grid crossed with depths 4/6 and child
weights 5/10/20 would recover 1,530/1,532 overall and 908/910 in the dense sweep.
It would exchange the two 0.10 winners for depth-6 child weights 10 and 20,
which had no archived wins. The observed replay measures inner-validation
selection. The hosted matched encoding run measures the prospective result
and runtime.

The matched 33-qubit launcher enables the reduced grid with
`PQFM_SIH_HYBRID_GRID=six`. The classical grid remains independent. The journal
records candidate order for each branch so candidate indices remain
interpretable and a resume checks that the search definition matches.
