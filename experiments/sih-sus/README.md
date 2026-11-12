# SIH-SUS Heisenberg MoE sweep

Open `sih_sus_reference_moe_sweep.ipynb` in JupyterLab to inspect or run the experiment. The notebook uses the frozen SIH-SUS admissions data and protocol under `data/sih-sus/protocolo_comparacao`, all available training rows, ten monthly test folds, and a TileLang Heisenberg expert. It fits a soft router using historical Analysis months and saves predictions and fitted models under `outputs/reference_soft_moe_sweep`.

The 2025 test months were previously inspected, so results are exploratory. The SIH-SUS test folds are one month each; the SIA-SUS PDF used six-month test folds. This notebook does not implement the PDF's separate no-Analysis baseline.

The sweep implementation is contained in the notebook and uses the shared `pqfmlib` TileLang backend.
