"""Past-only features and target-time splits; no random time-series split."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DATA_URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
REFERENCE_SHA256 = "2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d"
TARGET = "Appliances"
SENSORS = ["lights"] + [v for i in range(1, 10) for v in (f"T{i}", f"RH_{i}")]
SENSORS += ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
FEATURES = [TARGET] + SENSORS + ["hour_sin", "hour_cos", "week_sin", "week_cos"]


def download(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "energy-forecast-lab/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        matches = [n for n in z.namelist() if Path(n).name == "energydata_complete.csv"]
        if len(matches) != 1:
            raise ValueError("UCI archive must contain exactly one expected CSV")
        content = z.read(matches[0])
    if hashlib.sha256(content).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Dataset bytes differ from the reference; review the UCI release before updating its hash")
    path = destination / "energydata_complete.csv"
    temporary = destination / "energydata_complete.csv.part"
    temporary.write_bytes(content)
    frame = read_frame(temporary)
    if len(frame) != 19735:
        raise ValueError("Unexpected UCI row count; inspect data before training")
    temporary.replace(path)
    metadata = {"source": DATA_URL, "license": "CC BY 4.0", "rows": len(frame),
                "sha256": hashlib.sha256(content).hexdigest(),
                "start": str(frame.date.iloc[0]), "end": str(frame.date.iloc[-1])}
    (destination / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def read_frame(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = ["date", TARGET] + SENSORS
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    validate_frame(frame)
    return frame


def validate_frame(frame: pd.DataFrame) -> None:
    if len(frame) < 2 or frame.date.isna().any():
        raise ValueError("At least two valid timestamps are required")
    if not (frame.date.diff().iloc[1:] == pd.Timedelta(minutes=10)).all():
        raise ValueError("Rows must be ordered, unique and spaced exactly 10 minutes apart")
    values = frame[[TARGET] + SENSORS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Sensor values must be finite; missing values are not silently filled")
    if (frame[TARGET] < 0).any():
        raise ValueError("Appliance energy must be nonnegative")


def feature_matrix(frame: pd.DataFrame, univariate: bool = False) -> np.ndarray:
    if univariate:
        return frame[[TARGET]].to_numpy(dtype=np.float32)
    result = frame[[TARGET] + SENSORS].copy()
    minutes = frame.date.dt.hour * 60 + frame.date.dt.minute
    week = frame.date.dt.dayofweek + minutes / 1440
    result["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
    result["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)
    result["week_sin"] = np.sin(2 * np.pi * week / 7)
    result["week_cos"] = np.cos(2 * np.pi * week / 7)
    return result[FEATURES].to_numpy(dtype=np.float32)


def fit_scaler(features: np.ndarray, train_end: int) -> dict:
    history = features[:train_end].astype(np.float64)
    mean = history.mean(axis=0)
    scale = history.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist(), "fit_rows": int(train_end)}


def transform(features: np.ndarray, scaler: dict) -> np.ndarray:
    return ((features - np.asarray(scaler["mean"])) /
            np.asarray(scaler["scale"])).astype(np.float32)


def split_origins(n: int, context: int, horizon: int) -> tuple[dict, dict]:
    if context < 1 or horizon < 1:
        raise ValueError("Context and horizon must be positive")
    train_end, val_end = int(n * .70), int(n * .85)
    # Origin is the first target row, exclusive end of the observed context.
    # All horizon labels must be entirely contained within one partition.
    origins = {
        "train": np.arange(context, train_end - horizon + 1),
        "validation": np.arange(max(context, train_end), val_end - horizon + 1),
        "test": np.arange(max(context, val_end), n - horizon + 1),
    }
    if any(len(v) == 0 for v in origins.values()):
        raise ValueError("Not enough rows for three partitions with this context/horizon")
    return origins, {"train_end": train_end, "validation_end": val_end, "rows": n}


class Windows(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, origins: np.ndarray,
                 context: int, horizon: int):
        self.x, self.y, self.origins = x, y, origins
        self.context, self.horizon = context, horizon

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        origin = int(self.origins[index])
        return (torch.from_numpy(self.x[origin-self.context:origin].copy()),
                torch.from_numpy(self.y[origin:origin+self.horizon].copy()))


def ridge_features(x: np.ndarray, origins: np.ndarray, context: int) -> np.ndarray:
    # Same information cutoff as the neural model. No future weather is consumed.
    return np.stack([np.concatenate([x[o-1], x[o-min(6, context):o].mean(axis=0),
                                    x[o-context:o].mean(axis=0),
                                    x[o-context:o, :1].std(axis=0)]) for o in origins])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and validate the public UCI dataset")
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    print(download(parser.parse_args().output))
