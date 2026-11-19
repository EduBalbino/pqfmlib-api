"""Run with the stock and patched XGBoost v3.2.0 packages in separate processes."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import cupy as cp
import numpy as np
from sklearn.metrics import average_precision_score
import xgboost as xgb


def run(rows, columns, rounds, missing):
    rng = np.random.default_rng(641)
    train = rng.standard_normal((rows, columns), dtype=np.float32)
    valid = rng.standard_normal((rows // 2, columns), dtype=np.float32)
    y_train = train[:, 0] - train[:, 1] * train[:, 2] + train[:, 3] > 0
    y_valid = valid[:, 0] - valid[:, 1] * valid[:, 2] + valid[:, 3] > 0
    if missing:
        train[rng.random(train.shape) < 0.4] = np.nan
        valid[rng.random(valid.shape) < 0.4] = np.nan
    train_gpu, valid_gpu = cp.asarray(train), cp.asarray(valid)
    params = {
        "device": "cuda",
        "tree_method": "hist",
        "objective": "binary:logistic",
        "max_depth": 8,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "nthread": 2,
        "seed": 53,
    }
    label = cp.asarray(y_train, dtype=cp.float32)
    train_quantile = xgb.QuantileDMatrix(train_gpu, label=label, nthread=2)
    begin = time.perf_counter()
    ranker = xgb.train(params, train_quantile, rounds)
    ranking_fit = time.perf_counter() - begin
    raw = xgb.DMatrix(train_gpu, nthread=2)
    values = ranker.predict(raw, pred_contribs=True)
    scores = np.abs(values[:, :-1]).sum(axis=0, dtype=np.float64)
    mask = np.argsort(-scores, kind="stable")[:min(80, columns)]
    selected = xgb.QuantileDMatrix(train_gpu[:, mask], label=label, nthread=2)
    begin = time.perf_counter()
    model = xgb.train(params, selected, rounds)
    selected_fit = time.perf_counter() - begin
    preds = cp.asnumpy(model.inplace_predict(valid_gpu[:, mask]))
    report = {
        "missing": missing, "rows": rows, "columns": columns, "rounds": rounds,
        "ranking_sha256": hashlib.sha256(ranker.save_raw()).hexdigest(),
        "selected_sha256": hashlib.sha256(model.save_raw()).hexdigest(),
        "ordered_mask": mask.tolist(),
        "prediction_sha256": hashlib.sha256(preds.tobytes()).hexdigest(),
        "validation_ap": float(average_precision_score(y_valid, preds)),
    }
    print(json.dumps({"ranking_s": ranking_fit, "selected_s": selected_fit, **report}))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--rows", type=int, default=3072)
    parser.add_argument("--columns", type=int, default=96)
    parser.add_argument("--rounds", type=int, default=25)
    args = parser.parse_args()
    if xgb.__version__ != "3.2.0" or not xgb.build_info()["USE_CUDA"]:
        raise RuntimeError("Use a CUDA build of XGBoost 3.2.0")
    result = [run(args.rows, args.columns, args.rounds, missing) for missing in (False, True)]
    args.result.write_text(json.dumps(result, indent=2) + "\n")
    if args.reference is not None:
        reference = json.loads(args.reference.read_text())
        for expected, actual in zip(reference, result, strict=True):
            differences = [key for key in expected if expected[key] != actual[key]]
            if differences:
                raise AssertionError(f"missing={expected['missing']}, differences={differences}")
        print("Exact model, ordered mask, probability and AP parity for both inputs.")


if __name__ == "__main__":
    main()
