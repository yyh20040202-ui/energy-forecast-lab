from contextlib import asynccontextmanager
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .inference import Forecaster
from .data import SENSORS


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[dict] = Field(min_length=1, max_length=10000)


def create_app(checkpoint: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = checkpoint or Path(os.environ.get("ENERGY_CHECKPOINT", "artifacts/tcn.pt"))
        app.state.forecaster = Forecaster(path) if path.is_file() else None
        yield

    application = FastAPI(title="Energy Forecast Lab", version="0.1.0", lifespan=lifespan)

    @application.get("/health")
    def health():
        forecaster = application.state.forecaster
        return {"status": "ready" if forecaster else "model_missing"}

    @application.get("/schema")
    def schema():
        forecaster = application.state.forecaster
        if forecaster is None:
            raise HTTPException(503, "Run python -m energy_forecast.train first")
        return {"context": forecaster.context, "horizon": forecaster.horizon,
                "step_minutes": 10, "model_features": forecaster.features,
                "required_record_fields": ["date", "Appliances"] +
                ([] if forecaster.payload["univariate"] else SENSORS),
                "timestamp": "date; local timezone-naive, sorted at 10-minute intervals"}

    @application.post("/forecast")
    def forecast(request: ForecastRequest):
        forecaster = application.state.forecaster
        if forecaster is None:
            raise HTTPException(503, "Run python -m energy_forecast.train first")
        try:
            return forecaster.forecast(request.records)
        except (ValueError, OverflowError) as exc:
            raise HTTPException(422, str(exc)) from exc

    return application


app = create_app()
