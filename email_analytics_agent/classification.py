"""Runtime adapter for the trained scikit-learn CTD classification model."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from .models import Prediction
from .taxonomy import CTD_BY_DEPARTMENT
from .training import (
    ARTIFACT_SCHEMA_VERSION,
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_METRICS_PATH,
    EXPECTED_LABELS,
    MODEL_VERSION,
    calculate_sha256,
)


class ModelArtifactError(RuntimeError):
    """Signal that a trained model artifact cannot be trusted or served."""


class CTDClassifier(Protocol):
    """Define the prediction contract consumed by the LangGraph manager."""

    def predict(self, redacted_text: str) -> Prediction:
        """Return a CTD prediction and model-derived confidence."""


class SklearnCTDClassifier:
    """Load a validated calibrated text model and return probabilities.

    The class never trains or falls back to heuristics at runtime. A missing,
    corrupt, incompatible, or untrusted artifact fails closed with an explicit
    command for rebuilding it from the versioned training corpus.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_ARTIFACT_PATH,
        metrics_path: str | Path = DEFAULT_METRICS_PATH,
    ) -> None:
        """Verify sidecar metadata before loading the trusted local joblib model."""

        self.model_path = Path(model_path)
        self.metrics_path = Path(metrics_path)
        self._metadata = self._load_metrics()
        self._model, self._pipeline = self._load_estimators()

    @staticmethod
    def _training_hint() -> str:
        """Return the exact local command that creates the expected artifacts."""

        return ".\\.venv\\Scripts\\python.exe scripts\\train_model.py"

    def _load_metrics(self) -> dict[str, Any]:
        """Load and validate trusted metadata before unpickling the artifact."""

        if not self.model_path.is_file() or not self.metrics_path.is_file():
            raise ModelArtifactError(
                f"trained CTD model artifacts are missing; run: {self._training_hint()}"
            )
        try:
            metadata = json.loads(self.metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ModelArtifactError("model metrics sidecar is unreadable or invalid") from error
        if metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ModelArtifactError("model artifact schema is incompatible")
        if metadata.get("model_version") != MODEL_VERSION:
            raise ModelArtifactError("model version is incompatible")
        expected_environment = metadata.get("environment", {})
        runtime_environment = {
            "python_major_minor": ".".join(platform.python_version().split(".")[:2]),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        }
        mismatches = {
            name: (expected_environment.get(name), version)
            for name, version in runtime_environment.items()
            if expected_environment.get(name) != version
        }
        if mismatches:
            raise ModelArtifactError(
                f"runtime differs from the training environment {mismatches}; retrain the model"
            )
        if metadata.get("artifact_sha256") != calculate_sha256(self.model_path):
            raise ModelArtifactError("model artifact hash does not match its trusted sidecar")
        return metadata

    def _load_estimators(self) -> tuple[CalibratedClassifierCV, Pipeline]:
        """Load the hash-verified calibrated model and its fitted base pipeline."""

        try:
            bundle = joblib.load(self.model_path)
        except Exception as error:
            raise ModelArtifactError("trained CTD model artifact could not be loaded") from error
        if not isinstance(bundle, dict):
            raise ModelArtifactError("model artifact bundle has an invalid type")
        if bundle.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ModelArtifactError("model artifact bundle schema is incompatible")
        if bundle.get("model_version") != MODEL_VERSION:
            raise ModelArtifactError("model artifact bundle version is incompatible")
        model = bundle.get("model")
        pipeline = bundle.get("base_pipeline")
        if not isinstance(model, CalibratedClassifierCV) or not hasattr(model, "predict_proba"):
            raise ModelArtifactError("artifact does not contain a calibrated probability model")
        if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "predict_proba"):
            raise ModelArtifactError("artifact does not contain a fitted base text pipeline")
        if tuple(sorted(str(label) for label in model.classes_)) != EXPECTED_LABELS:
            raise ModelArtifactError("model artifact does not contain the complete CTD taxonomy")
        return model, pipeline

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Expose immutable top-level training metadata for observability."""

        return MappingProxyType(self._metadata)

    @property
    def pipeline(self) -> Pipeline:
        """Expose the fitted word/character base pipeline for diagnostics."""

        return self._pipeline

    @property
    def model(self) -> CalibratedClassifierCV:
        """Expose the separately calibrated serving estimator for diagnostics."""

        return self._model

    def probability_distribution(self, redacted_text: str) -> dict[str, float]:
        """Return calibrated class probabilities for already-redacted email text."""

        probabilities = self._model.predict_proba([redacted_text])[0]
        return {
            str(label): float(probability)
            for label, probability in zip(self._model.classes_, probabilities, strict=True)
        }

    def predict(self, redacted_text: str) -> Prediction:
        """Return the highest-probability CTD and exact calibrated confidence."""

        distribution = self.probability_distribution(redacted_text)
        ranking = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
        department, confidence = ranking[0]
        runner_up, runner_up_confidence = ranking[1]
        return Prediction(
            CTD_BY_DEPARTMENT[department],
            round(confidence, 6),
            MODEL_VERSION,
            (
                f"Calibrated ML probability {confidence:.4f} for {department}; "
                f"runner-up {runner_up} scored {runner_up_confidence:.4f}."
            ),
        )

