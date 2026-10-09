"""An offline integration test covers training, export and real checkpoint loading."""
import json

import numpy as np
import pandas as pd

from energy_forecast.data import SENSORS
from energy_forecast.inference import Forecaster
from energy_forecast.train import run


def test_full_training_and_export(tmp_path):
    rows = 320
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"date": pd.date_range("2016-01-01", periods=rows, freq="10min"),
                         "Appliances": 80 + 20*np.sin(np.arange(rows)/10),
                         **{key: rng.normal(20, 1, rows) for key in SENSORS}})
    data = tmp_path / "synthetic.csv"
    frame.to_csv(data, index=False)
    output = tmp_path / "run"
    config = {"context": 36, "horizon": 6, "channels": 4,
              "seed": 42, "epochs": 1, "patience": 1}
    results = run(data, output, config)
    assert results["data"]["rows"] == rows
    assert set(results["test"]) == {"Persistence", "Ridge", "TCN", "TCN-target-only"}
    assert results["scaler_fit_rows"] == int(rows*.7)
    request = json.loads((output / "examples/request.json").read_text())
    forecast = Forecaster(output / "artifacts/tcn.pt").forecast(request["records"])
    assert len(forecast["predictions"]) == 6
    assert all(np.isfinite(p["energy_wh"]) and p["energy_wh"] >= 0 for p in forecast["predictions"])
    assert (output / "reports/benchmark.png").stat().st_size > 0
