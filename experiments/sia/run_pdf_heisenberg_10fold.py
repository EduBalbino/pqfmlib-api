"""Reproduce the PDF's isolated Heisenberg expert protocol across ten folds.

This runner reuses the SIA notebook's data checks, temporal folds, expert
training, AP tuning, and exports. It runs only the no-Analysis comparison and
pins the Heisenberg settings to the October results PDF.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[2]
SIA_DIR = ROOT / "experiments" / "sia"
SOURCE_NOTEBOOK = SIA_DIR / "QIMED_MoE_Demanda_Pressao_pqfmlib_api.ipynb"
DATASET = ROOT / "data" / "sia-sus" / "protocolo_comparacao" / "base_modelagem_congelada.parquet"
PDF = SIA_DIR / "heisenberg_expert.pdf"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one notebook setting: {old!r}")
    return source.replace(old, new, 1)


def prepare_run_notebook():
    source_nb = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    source_indexes = [index for index in range(2, 53) if index != 45]
    cells = [copy.deepcopy(source_nb.cells[index]) for index in source_indexes]
    by_source_index = dict(zip(source_indexes, cells))

    config = by_source_index[23]
    config.source = replace_once(config.source, "n_components=75, q_enc=26", "n_components=54, q_enc=19")
    config.source = replace_once(config.source, "hybrid_shap_top_k=80", "hybrid_shap_top_k=50")
    config.source = replace_once(config.source, "R=3, alpha=0.2", "R=2, alpha=0.1")

    benchmark = by_source_index[39]
    benchmark.source = replace_once(benchmark.source, "HEISENBERG_BENCHMARK_SAMPLES = 32",
                                    "HEISENBERG_BENCHMARK_SAMPLES = 2")
    benchmark.source = replace_once(benchmark.source, "HEISENBERG_BENCHMARK_REPEATS = 3",
                                    "HEISENBERG_BENCHMARK_REPEATS = 1")

    moe = by_source_index[41]
    moe.source = replace_once(moe.source, "RUN_CLASSIFICATION_MOE = True", "RUN_CLASSIFICATION_MOE = False")
    analysis = by_source_index[44]
    analysis.source = replace_once(analysis.source, "ANALYZE_CLASSIFICATION = True",
                                   "ANALYZE_CLASSIFICATION = False")
    summarize = by_source_index[52]
    summarize.source = replace_once(summarize.source, "ANALYZE_NO_ANALYSIS = True",
                                    "ANALYZE_NO_ANALYSIS = False")

    report = nbformat.v4.new_code_cell(
        """import json

metrics = classificacao_sem_analysis_results['metrics'].copy()
ap = metrics.pivot(index='fold', columns='scenario', values='AP')
ap['heisenberg_minus_classic'] = ap['heisenberg_sem_analysis'] - ap['classico_sem_analysis']
print(metrics[['fold', 'scenario', 'AP', 'ROC-AUC', 'F1', 'tuning_seconds', 'refit_seconds', 'inference_seconds']].to_string(index=False))
print('\\nFold-level AP comparison:')
print(ap.round(6).to_string())
print('\\nSummary:', json.dumps({
    'folds': int(len(ap)),
    'classic_mean_ap': float(ap['classico_sem_analysis'].mean()),
    'heisenberg_mean_ap': float(ap['heisenberg_sem_analysis'].mean()),
    'mean_ap_gain': float(ap['heisenberg_minus_classic'].mean()),
    'folds_won': int((ap['heisenberg_minus_classic'] > 0).sum()),
    'folds_total': int(len(ap)),
    'classic_mean_tuning_seconds': float(metrics.loc[metrics.scenario.eq('classico_sem_analysis'), 'tuning_seconds'].mean()),
    'heisenberg_mean_tuning_seconds': float(metrics.loc[metrics.scenario.eq('heisenberg_sem_analysis'), 'tuning_seconds'].mean()),
    'heisenberg_mean_refit_seconds': float(metrics.loc[metrics.scenario.eq('heisenberg_sem_analysis'), 'refit_seconds'].mean()),
}, indent=2))
"""
    )
    # The classically selected hyperparameters and threshold are tuned on each
    # fold's historical six-month validation window; outer tests remain untouched
    # during selection. This matches the report's isolated-expert comparison.
    cells.append(report)

    run_nb = nbformat.v4.new_notebook(cells=cells, metadata=copy.deepcopy(source_nb.metadata))
    return run_nb


def main() -> None:
    for path in (SOURCE_NOTEBOOK, DATASET, PDF):
        if not path.is_file():
            raise FileNotFoundError(path)
    started = time.perf_counter()
    run_nb = prepare_run_notebook()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact = SIA_DIR / "outputs" / f"pdf_heisenberg_10fold_{stamp}.ipynb"
    artifact.parent.mkdir(parents=True, exist_ok=True)

    client = NotebookClient(
        run_nb,
        timeout=3600,
        kernel_name=run_nb.metadata.get("kernelspec", {}).get("name", "python3"),
        resources={"metadata": {"path": str(SIA_DIR)}},
        allow_errors=False,
    )
    executed = client.execute()
    nbformat.write(executed, artifact)

    print(f"Execution notebook: {artifact}")
    print(f"Dataset SHA-256: {hashlib.sha256(DATASET.read_bytes()).hexdigest()}")
    print(f"PDF SHA-256: {hashlib.sha256(PDF.read_bytes()).hexdigest()}")
    print(f"Elapsed seconds: {time.perf_counter() - started:.3f}")

    for cell in executed.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type == "stream" and "Summary:" in output.text:
                print(output.text[-2000:])


if __name__ == "__main__":
    main()
