"""Checkpoint inference shares exactly the training feature/scaling pipeline."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import feature_matrix, transform, validate_frame, SENSORS
from .model import TCN


class Forecaster:
    def __init__(self, checkpoint: Path | str):
        # Only load checkpoints produced locally/by trusted project authors.
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported checkpoint schema")
        self.payload = payload
        self.context, self.horizon = payload["context"], payload["horizon"]
        self.features = payload["features"]
        self.model = TCN(**payload["model_config"])
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def forecast(self, records: list[dict]) -> dict:
        if len(records) != self.context:
            raise ValueError(f"Expected exactly {self.context} history rows")
        frame = pd.DataFrame(records)
        expected = ["date", "Appliances"] + ([] if self.payload["univariate"] else SENSORS)
        if set(expected) - set(frame.columns):
            raise ValueError(f"Every record must contain {expected}")
        try:
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            if frame.date.dt.tz is not None:
                raise ValueError("Use timezone-naive timestamps in the dataset's local time")
            if self.payload["univariate"]:
                for key in SENSORS:
                    frame[key] = 0.0
            validate_frame(frame)
            x = feature_matrix(frame, univariate=self.payload["univariate"])
        except (TypeError, KeyError) as exc:
            raise ValueError("Invalid timestamp or numeric sensor value") from exc
        scaler = self.payload["scaler"]
        x = transform(x, scaler)
        with torch.inference_mode():
            prediction = self.model(torch.from_numpy(x).unsqueeze(0)).numpy()[0]
        prediction = np.maximum(0, prediction * scaler["scale"][0] + scaler["mean"][0])
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite forecast; check input magnitudes")
        timestamps = pd.date_range(frame.date.iloc[-1] + pd.Timedelta(minutes=10),
                                   periods=self.horizon, freq="10min")
        return {"model": "TCN-target-only" if self.payload["univariate"] else "TCN",
                "unit": "Wh per 10-minute interval", "step_minutes": 10,
                "predictions": [{"date": str(t), "energy_wh": float(v)} for t, v in zip(timestamps, prediction)]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/tcn.pt"))
    parser.add_argument("--input", type=Path, default=Path("examples/request.json"))
    args = parser.parse_args()
    result = Forecaster(args.checkpoint).forecast(json.loads(args.input.read_text(encoding="utf-8"))["records"])
    print(json.dumps(result, indent=2))
