"""Used-car price predictor helper with feature adaptation.

Workflow summary:

1. Train a regression pipeline offline (RandomForest, XGBoost, etc.) using the
   Kaggle dataset; persist it with ``joblib.dump``.
2. Place the artefact at ``car_analysis/models/used_car_price_xgb_pipeline.joblib``
   or point ``CAR_ML_MODEL_PATH`` to a custom path.
3. Call :func:`ml_predictor.predict_price` with the raw car attributes
   (year/mileage/fuel_type/...).  The helper will convert them into the feature
   schema expected by the model (Vehicle_Age, Mileage_per_Year, etc.) before
   invoking the pipeline.

The adapter assumes the training target was ``log1p(price)`` and therefore
applies ``np.expm1`` to recover the final dollar estimate. Adjust if your model
uses a different target transformation.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

try:
    import joblib
except Exception:  # pragma: no cover - optional dependency
    joblib = None  # type: ignore

import numpy as np


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "used_car_price_xgb_pipeline.joblib"
)


def feature_adapter(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt raw input fields into the engineered feature schema.

    Parameters
    ----------
    raw:
        Dictionary with user-provided attributes. Typical keys include ``year``,
        ``mileage``, ``hp``, ``engine_displacement``, ``fuel_type`` etc.

    Returns
    -------
    Dict[str, Any]
        Feature dict suitable for the persisted sklearn pipeline.
    """

    current_year = datetime.utcnow().year
    year = raw.get("year")
    mileage = raw.get("mileage")

    vehicle_age = 0
    if isinstance(year, (int, float)) and year > 0:
        vehicle_age = max(current_year - int(year), 0)

    if isinstance(mileage, (int, float)) and vehicle_age > 0:
        mileage_per_year = float(mileage) / vehicle_age
    elif isinstance(mileage, (int, float)):
        mileage_per_year = float(mileage)
    else:
        mileage_per_year = 0.0

    return {
        "hp": raw.get("hp", 0.0),
        "engine displacement": raw.get("engine_displacement", 0.0),
        "Vehicle_Age": vehicle_age,
        "Mileage_per_Year": mileage_per_year,
        "fuel_type": raw.get("fuel_type", "OTHER"),
        "transmission": raw.get("transmission", "OTHER"),
        "is_v_engine": raw.get("is_v_engine", 0),
        "clean_title": raw.get("clean_title", 0),
    }


class PricePredictor:
    """Wrapper around a scikit-learn regression pipeline."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(
            model_path or os.getenv("CAR_ML_MODEL_PATH", str(DEFAULT_MODEL_PATH))
        )
        self._pipeline: Optional[Any] = None

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the persisted pipeline from disk if available."""

        if self._pipeline is not None:
            return

        if joblib is None:
            raise RuntimeError(
                "joblib is not installed. Install optional dependencies first."
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Set CAR_ML_MODEL_PATH."
            )

        logger.info("Loading ML predictor from %s", self.model_path)
        self._pipeline = joblib.load(self.model_path)

    @property
    def available(self) -> bool:
        """Whether the predictor can be used."""

        try:
            self.load()
            return True
        except Exception as exc:
            logger.debug("Predictor unavailable: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Prediction API
    # ------------------------------------------------------------------

    def predict_price(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Predict used car price given feature dict.

        The `features` dict should align with what the training pipeline expects.
        A minimal example (you can add more engineered features):

        ```python
        features = {
            "year": 2019,
            "mileage": 42000,
            "make": "Toyota",
            "model": "Camry",
            "state": "CA",
            "trim": "SE",
            "fuel": "Gasoline",
        }
        ```
        """

        if self._pipeline is None:
            self.load()

        assert self._pipeline is not None

        adapted_features = feature_adapter(features)
        # Most sklearn pipelines accept a list of dicts (converted to DataFrame).
        try:
            prediction: Iterable[float] = self._pipeline.predict([adapted_features])
        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            raise

        raw_pred = float(list(prediction)[0])
        predicted_price = float(np.expm1(raw_pred))
        return {
            "success": True,
            "predicted_price": predicted_price,
            "model_path": str(self.model_path),
            "features_used": adapted_features,
        }


# Singleton predictor for quick access
ml_predictor = PricePredictor()
