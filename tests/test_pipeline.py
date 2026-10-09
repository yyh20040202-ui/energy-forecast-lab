import json

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
import torch

from energy_forecast.api import create_app
from energy_forecast.data import (FEATURES, SENSORS, Windows, feature_matrix,
                                  fit_scaler, split_origins, validate_frame)
from energy_forecast.inference import Forecaster
from energy_forecast.metrics import metrics
from energy_forecast.model import CausalConv, TCN


def sample_frame(rows=200):
    return pd.DataFrame({"date": pd.date_range("2016-01-01", periods=rows, freq="10min"),
                         "Appliances": np.arange(rows, dtype=float) + 1,
                         **{key: np.ones(rows) for key in SENSORS}})


def checkpoint(tmp_path):
    config = {"input_channels": len(FEATURES), "horizon": 6, "channels": 4, "dropout": 0.}
    path = tmp_path / "test.pt"
    torch.save({"schema_version": 1, "model_config": config,
                "state_dict": TCN(**config).state_dict(), "context": 36, "horizon": 6,
                "features": FEATURES, "univariate": False,
                "scaler": {"mean": [0.] * len(FEATURES), "scale": [1.] * len(FEATURES)}}, path)
    return path


def records():
    frame = sample_frame(36)
    frame["date"] = frame.date.astype(str)
    return frame.to_dict("records")


def test_partition_targets_never_overlap():
    origins, bounds = split_origins(200, 36, 6)
    assert origins["train"][-1] + 6 <= bounds["train_end"]
    assert origins["validation"][0] >= bounds["train_end"]
    assert origins["validation"][-1] + 6 <= bounds["validation_end"]
    assert origins["test"][0] >= bounds["validation_end"]


def test_window_excludes_target_and_future():
    x = np.arange(200, dtype=np.float32)[:, None]
    dataset = Windows(x, x[:, 0], np.array([100]), 36, 6)
    history, target = dataset[0]
    assert history[-1, 0] == 99
    assert target.tolist() == [100, 101, 102, 103, 104, 105]
    x[100:] = 9999
    assert torch.equal(dataset[0][0], history)


def test_scaler_ignores_holdout_values():
    x = np.arange(200, dtype=np.float32)[:, None]
    first = fit_scaler(x, 140)
    x[140:] = 100000
    assert first == fit_scaler(x, 140)


def test_random_columns_are_excluded():
    frame = sample_frame()
    frame["rv1"], frame["rv2"] = 42., 42.
    assert feature_matrix(frame).shape[1] == 30
    assert "rv1" not in FEATURES and "rv2" not in FEATURES


@pytest.mark.parametrize("issue", ["duplicate", "gap", "nan", "negative"])
def test_bad_data_is_rejected(issue):
    frame = sample_frame()
    if issue == "duplicate":
        frame.loc[1, "date"] = frame.loc[0, "date"]
    elif issue == "gap":
        frame.loc[1, "date"] += pd.Timedelta(minutes=1)
    elif issue == "nan":
        frame.loc[1, "T1"] = np.nan
    else:
        frame.loc[1, "Appliances"] = -1
    with pytest.raises(ValueError):
        validate_frame(frame)


def test_causal_conv_cannot_see_future():
    torch.manual_seed(42)
    layer = CausalConv(2, 4, 2).eval()
    x = torch.randn(1, 2, 36)
    y = layer(x)
    x[..., 20:] += 1000
    torch.testing.assert_close(layer(x)[..., :20], y[..., :20])


def test_metrics_zero_and_shape():
    result = metrics(np.zeros((2, 6)), np.zeros((2, 6)))
    assert result == {"mae_wh": 0., "rmse_wh": 0., "smape_percent": 0.}
    with pytest.raises(ValueError):
        metrics(np.ones((1, 6)), np.ones((1, 5)))


def test_checkpoint_and_api_agree(tmp_path):
    path = checkpoint(tmp_path)
    expected = Forecaster(path).forecast(records())
    assert len(expected["predictions"]) == 6
    assert expected["predictions"][0]["date"] == "2016-01-01 06:00:00"
    with TestClient(create_app(path)) as client:
        assert client.get("/health").json()["status"] == "ready"
        response = client.post("/forecast", json={"records": records()})
        assert response.status_code == 200
        assert response.json() == expected
        assert client.post("/forecast", json={"records": records()[:2]}).status_code == 422
        malformed = records()
        malformed[0]["T1"] = "invalid"
        assert client.post("/forecast", json={"records": malformed}).status_code == 422


def test_missing_checkpoint_returns_503(tmp_path):
    with TestClient(create_app(tmp_path / "missing.pt")) as client:
        assert client.get("/health").json()["status"] == "model_missing"
        assert client.post("/forecast", json={"records": records()}).status_code == 503
