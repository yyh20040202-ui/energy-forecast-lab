"""Fixed-seed experiments. Validation selects checkpoints; test is report-only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import sklearn
import torch
from torch.utils.data import DataLoader

from .data import (FEATURES, REFERENCE_SHA256, Windows, feature_matrix, fit_scaler, read_frame,
                   ridge_features, split_origins, transform)
from .metrics import metrics
from .model import TCN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def predict(model: TCN, dataset: Windows) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(x) for x, _ in DataLoader(dataset, batch_size=256)]).numpy()


def train_tcn(frame: pd.DataFrame, origins: dict, bounds: dict, config: dict,
              univariate: bool = False) -> tuple[dict, dict, list]:
    seed_everything(config["seed"])
    raw_x = feature_matrix(frame, univariate=univariate)
    scaler = fit_scaler(raw_x, bounds["train_end"])
    x = transform(raw_x, scaler)
    y = ((frame.Appliances.to_numpy(dtype=np.float32) - scaler["mean"][0]) /
         scaler["scale"][0]).astype(np.float32)
    windows = {name: Windows(x, y, rows, config["context"], config["horizon"])
               for name, rows in origins.items()}
    model_config = {"input_channels": x.shape[1], "horizon": config["horizon"],
                    "channels": config["channels"], "dropout": .1}
    model = TCN(**model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    best_state = None
    logs = []
    loader = DataLoader(windows["train"], batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(config["seed"]))
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = predict(model, windows["validation"])
        actual = np.stack([y[o:o+config["horizon"]] for o in origins["validation"]])
        val_loss = float(np.mean((validation - actual) ** 2))
        logs.append({"epoch": epoch, "train_mse": float(np.mean(losses)),
                     "validation_mse": val_loss})
        print(f"{'TCN-target-only' if univariate else 'TCN'} epoch {epoch}: val MSE={val_loss:.4f}", flush=True)
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    outputs = {name: np.maximum(0, predict(model, windows[name]) * scaler["scale"][0] +
                                scaler["mean"][0]) for name in ("validation", "test")}
    checkpoint = {"schema_version": 1, "state_dict": best_state, "model_config": model_config,
                  "scaler": scaler, "context": config["context"], "horizon": config["horizon"],
                  "features": ["Appliances"] if univariate else FEATURES,
                  "univariate": univariate, "best_epoch": best_epoch,
                  "seed": config["seed"]}
    return outputs, checkpoint, logs


def run(data: Path, output: Path, config: dict) -> dict:
    if config["epochs"] < 1 or config["patience"] < 1 or config["channels"] < 1:
        raise ValueError("epochs/patience/channels must be positive")
    started = time.perf_counter()
    seed_everything(config["seed"])
    frame = read_frame(data)
    origins, bounds = split_origins(len(frame), config["context"], config["horizon"])
    output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "artifacts"
    reports = output / "reports"
    artifacts.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)
    y = frame.Appliances.to_numpy(dtype=np.float32)
    truth = {name: np.stack([y[o:o+config["horizon"]] for o in rows]) for name, rows in origins.items()}
    predictions = {name: {} for name in ("validation", "test")}
    for name in predictions:
        predictions[name]["Persistence"] = np.repeat(y[origins[name]-1, None], config["horizon"], axis=1)
    x = feature_matrix(frame)
    scaler = fit_scaler(x, bounds["train_end"])
    x = transform(x, scaler)
    design = {name: ridge_features(x, rows, config["context"]) for name, rows in origins.items()}
    best_ridge, best_alpha, best_score = None, None, float("inf")
    for alpha in (1., 10., 100., 1000.):
        model = Ridge(alpha=alpha).fit(design["train"], truth["train"])
        score = metrics(truth["validation"], np.maximum(0, model.predict(design["validation"])))
        if score["rmse_wh"] < best_score:
            best_ridge, best_alpha, best_score = model, alpha, score["rmse_wh"]
    for name in predictions:
        predictions[name]["Ridge"] = np.maximum(0, best_ridge.predict(design[name]))
    training_log = {}
    for univariate, label in ((False, "TCN"), (True, "TCN-target-only")):
        outputs, checkpoint, logs = train_tcn(frame, origins, bounds, config, univariate)
        torch.save(checkpoint, artifacts / ("tcn_target_only.pt" if univariate else "tcn.pt"))
        training_log[label] = logs
        for name in predictions:
            predictions[name][label] = outputs[name]
    results = {name: {label: metrics(truth[name], p) for label, p in models.items()}
               for name, models in predictions.items()}
    rows = []
    for name, models in predictions.items():
        for label, p in models.items():
            rows.append({"split": name, "model": label, **results[name][label]})
    pd.DataFrame(rows).to_csv(reports / "metrics.csv", index=False)
    horizon_results = {label: [metrics(truth["test"][:, i:i+1], p[:, i:i+1])
                               for i in range(config["horizon"])]
                       for label, p in predictions["test"].items()}
    cutoffs = {name: {"windows": len(indices), "first_target": str(frame.date.iloc[indices[0]]),
                      "last_target": str(frame.date.iloc[indices[-1]+config["horizon"]-1])}
               for name, indices in origins.items()}
    data_hash = hashlib.sha256(data.read_bytes()).hexdigest()
    source = ("UCI Appliances Energy Prediction, DOI 10.24432/C5VC8G"
              if data_hash == REFERENCE_SHA256 else "User-supplied CSV (not the reference UCI dataset)")
    manifest = {"schema_version": 1, "data": {"sha256": data_hash,
                "rows": len(frame), "source": source},
                "config": config, "partitions": cutoffs,
                "scaler_fit_rows": bounds["train_end"], "ridge_alpha": best_alpha,
                "validation": results["validation"], "test": results["test"],
                "test_per_horizon": horizon_results,
                "runtime_seconds": round(time.perf_counter()-started, 2),
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "numpy": np.__version__, "pandas": pd.__version__,
                                "sklearn": sklearn.__version__, "device": "cpu"},
                "selection_policy": "Ridge alpha and TCN checkpoints use validation only; test reported once",
                "evaluation": "rolling origin with newly observed true history; overlapping forecast labels"}
    (reports / "results.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (reports / "training_log.json").write_text(json.dumps(training_log, indent=2), encoding="utf-8")
    # A prediction example is public dataset information only.
    example_origin = int(origins["test"][0])
    example = {"records": [{"date": str(row.date), **{key: float(row[key]) for key in FEATURES[:-4]}}
                           for _, row in frame.iloc[example_origin-config["context"]:example_origin].iterrows()]}
    examples = output / "examples"
    examples.mkdir(exist_ok=True)
    (examples / "request.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    labels = list(results["test"])
    axes[0].bar(labels, [results["test"][key]["mae_wh"] for key in labels], color=["#b8c5d6", "#7993b5", "#2563a6", "#5a9b88"])
    axes[0].set(ylabel="MAE (Wh per 10-minute interval)", title="Chronological holdout, all 6 horizons")
    axes[0].tick_params(axis="x", labelrotation=18)
    steps = np.arange(1, config["horizon"]+1)*10
    for label in labels:
        axes[1].plot(steps, [v["mae_wh"] for v in horizon_results[label]], marker="o", label=label)
    axes[1].set(xlabel="Forecast horizon (minutes)", ylabel="MAE (Wh)", title="Error by forecast horizon")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(reports / "benchmark.png", dpi=160)
    plt.close(figure)
    forecast = pd.DataFrame({"target_time": frame.date.iloc[origins["test"]].astype(str).to_numpy(),
                             "actual_first_step_wh": truth["test"][:, 0],
                             **{label: p[:, 0] for label, p in predictions["test"].items()}})
    forecast.to_csv(reports / "test_predictions.csv", index=False)
    print(json.dumps(results["test"], indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPU benchmark and target-only ablation")
    parser.add_argument("--data", type=Path, default=Path("data/raw/energydata_complete.csv"))
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--context", type=int, default=36)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    configuration = {k: v for k, v in vars(args).items() if k not in ("data", "output")}
    run(args.data, args.output, configuration)
