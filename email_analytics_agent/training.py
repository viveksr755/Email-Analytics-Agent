"""Leakage-resistant training, calibration, evaluation, and persistence."""

from __future__ import annotations

import hashlib
import json
import platform
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from .models import EmailMessage
from .preprocessing import EmailPreprocessor
from .taxonomy import CTD_BY_DEPARTMENT


ARTIFACT_SCHEMA_VERSION = "2.0"
MODEL_VERSION = "sklearn-word-char-logreg-v2"
DEFAULT_DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "training_emails.json"
DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "ctd_classifier.joblib"
DEFAULT_METRICS_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "ctd_classifier.metrics.json"
EXPECTED_LABELS = tuple(sorted(CTD_BY_DEPARTMENT))
SUPERVISED_SPLITS = ("train", "calibration", "validation", "challenge")
ALL_SPLITS = (*SUPERVISED_SPLITS, "ood")
REGULARIZATION_GRID = (0.05, 0.10, 0.25, 0.50, 1.00)


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Contain PII-safe partitions and semantic groups used during training."""

    train_texts: tuple[str, ...]
    train_labels: tuple[str, ...]
    train_groups: tuple[str, ...]
    calibration_texts: tuple[str, ...]
    calibration_labels: tuple[str, ...]
    validation_texts: tuple[str, ...]
    validation_labels: tuple[str, ...]
    challenge_texts: tuple[str, ...]
    challenge_labels: tuple[str, ...]
    ood_texts: tuple[str, ...]
    families_by_split: Mapping[str, frozenset[str]]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Describe artifact locations and the complete training metadata record."""

    artifact_path: Path
    metrics_path: Path
    metadata: Mapping[str, Any]


def calculate_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a dataset or trusted local artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_rows(path: str | Path = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    """Load and structurally validate the labeled and OOD JSON corpus."""

    resolved = Path(path)
    values = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not values:
        raise ValueError("training dataset must be a non-empty JSON array")
    required = {
        "message_id",
        "sender",
        "subject",
        "body",
        "received_at",
        "expected_department",
        "split",
        "template_family",
    }
    for index, value in enumerate(values):
        if not isinstance(value, dict) or not required.issubset(value):
            raise ValueError(f"training row {index} is missing required fields")
        split = value["split"]
        if split not in ALL_SPLITS:
            raise ValueError(f"training row {index} has an unsupported split")
        expected = value["expected_department"]
        if split == "ood" and expected != "RM Review":
            raise ValueError(f"OOD row {index} must be marked for RM Review")
        if split != "ood" and expected not in EXPECTED_LABELS:
            raise ValueError(f"training row {index} has an unsupported department")
    return values


def prepare_training_data(
    rows: Iterable[Mapping[str, Any]],
    preprocessor: EmailPreprocessor | None = None,
) -> PreparedDataset:
    """Apply production redaction and reject duplicate or group-leaking splits."""

    processor = preprocessor or EmailPreprocessor()
    texts: dict[str, list[str]] = {split: [] for split in ALL_SPLITS}
    labels: dict[str, list[str]] = {split: [] for split in SUPERVISED_SPLITS}
    families: dict[str, set[str]] = {split: set() for split in ALL_SPLITS}
    train_groups: list[str] = []

    for row in rows:
        split = str(row["split"])
        email = EmailMessage.from_dict(row)
        safe_text = processor.process(email).model_text
        if processor.redactor.contains_pii(safe_text):
            raise ValueError(f"PII remained after preprocessing message {email.message_id}")
        family = str(row["template_family"])
        texts[split].append(safe_text)
        families[split].add(family)
        if split in SUPERVISED_SPLITS:
            labels[split].append(str(row["expected_department"]))
        if split == "train":
            train_groups.append(family)

    text_sets = {split: set(values) for split, values in texts.items()}
    for split in ALL_SPLITS:
        if len(text_sets[split]) != len(texts[split]):
            raise ValueError(f"duplicate redacted texts detected inside {split}")
    for left, right in combinations(ALL_SPLITS, 2):
        if text_sets[left] & text_sets[right]:
            raise ValueError(f"redacted text leakage detected between {left} and {right}")
        if families[left] & families[right]:
            raise ValueError(f"template-family leakage detected between {left} and {right}")

    for split in SUPERVISED_SPLITS:
        counts = Counter(labels[split])
        if set(counts) != set(EXPECTED_LABELS) or len(set(counts.values())) != 1:
            raise ValueError(f"{split} partition must be balanced across all CTD labels")
    if not texts["ood"]:
        raise ValueError("OOD partition must contain ambiguous or unsupported examples")

    return PreparedDataset(
        train_texts=tuple(texts["train"]),
        train_labels=tuple(labels["train"]),
        train_groups=tuple(train_groups),
        calibration_texts=tuple(texts["calibration"]),
        calibration_labels=tuple(labels["calibration"]),
        validation_texts=tuple(texts["validation"]),
        validation_labels=tuple(labels["validation"]),
        challenge_texts=tuple(texts["challenge"]),
        challenge_labels=tuple(labels["challenge"]),
        ood_texts=tuple(texts["ood"]),
        families_by_split={split: frozenset(values) for split, values in families.items()},
    )


def build_base_pipeline(random_seed: int = 42) -> Pipeline:
    """Build regularized word/character TF-IDF features and logistic regression."""

    features = FeatureUnion(
        transformer_list=[
            (
                "word",
                TfidfVectorizer(
                    lowercase=True,
                    max_df=0.98,
                    max_features=20_000,
                    min_df=2,
                    ngram_range=(1, 2),
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    lowercase=True,
                    max_features=30_000,
                    min_df=2,
                    ngram_range=(3, 5),
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(
        C=0.25,
        class_weight="balanced",
        max_iter=2_000,
        random_state=random_seed,
        solver="lbfgs",
    )
    return Pipeline(steps=[("features", features), ("classifier", classifier)])


def _expected_calibration_error(
    expected: tuple[str, ...],
    predicted: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute top-label expected calibration error over equal-width bins."""

    expected_values = np.asarray(expected)
    confidences = probabilities.max(axis=1)
    correct = predicted == expected_values
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidences > lower) & (confidences <= upper)
        if not selected.any():
            continue
        error += float(selected.mean()) * abs(
            float(correct[selected].mean()) - float(confidences[selected].mean())
        )
    return error


def _threshold_metrics(
    expected: tuple[str, ...],
    predicted: np.ndarray,
    probabilities: np.ndarray,
    selected_threshold: float,
) -> dict[str, dict[str, float | int]]:
    """Measure selective accuracy, coverage, and error risk by threshold."""

    expected_values = np.asarray(expected)
    confidences = probabilities.max(axis=1)
    thresholds = sorted({0.60, 0.70, 0.80, 0.90, round(selected_threshold, 2)})
    results: dict[str, dict[str, float | int]] = {}
    for threshold in thresholds:
        selected = confidences >= threshold
        count = int(selected.sum())
        selected_accuracy = (
            float(accuracy_score(expected_values[selected], predicted[selected])) if count else 0.0
        )
        results[f"{threshold:.2f}"] = {
            "count": count,
            "coverage": round(count / len(expected), 6),
            "selective_accuracy": round(selected_accuracy, 6),
            "selective_risk": round(1.0 - selected_accuracy, 6) if count else 0.0,
            "errors": int((predicted[selected] != expected_values[selected]).sum()) if count else 0,
        }
    return results


def _evaluate(
    model: CalibratedClassifierCV,
    texts: tuple[str, ...],
    expected: tuple[str, ...],
    label_order: tuple[str, ...],
    selected_threshold: float,
) -> dict[str, Any]:
    """Evaluate class quality, probability quality, and selective routing."""

    predicted = model.predict(texts)
    probabilities = model.predict_proba(texts)
    return {
        "count": len(expected),
        "accuracy": round(float(accuracy_score(expected, predicted)), 6),
        "macro_precision": round(
            float(precision_score(expected, predicted, average="macro", zero_division=0)), 6
        ),
        "macro_recall": round(
            float(recall_score(expected, predicted, average="macro", zero_division=0)), 6
        ),
        "macro_f1": round(float(f1_score(expected, predicted, average="macro")), 6),
        "log_loss": round(float(log_loss(expected, probabilities, labels=label_order)), 6),
        "brier_score": round(
            float(brier_score_loss(expected, probabilities, labels=label_order)), 6
        ),
        "expected_calibration_error": round(
            _expected_calibration_error(expected, predicted, probabilities), 6
        ),
        "mean_confidence": round(float(probabilities.max(axis=1).mean()), 6),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=EXPECTED_LABELS).tolist(),
        "classification_report": classification_report(
            expected,
            predicted,
            labels=EXPECTED_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "routing_thresholds": _threshold_metrics(
            expected, predicted, probabilities, selected_threshold
        ),
    }


def _select_routing_threshold(
    model: CalibratedClassifierCV,
    texts: tuple[str, ...],
    expected: tuple[str, ...],
    minimum_threshold: float,
    target_precision: float = 0.98,
) -> tuple[float, dict[str, float | int]]:
    """Select maximum-coverage calibration threshold satisfying precision policy."""

    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)
    confidences = probabilities.max(axis=1)
    expected_values = np.asarray(expected)
    for threshold in np.arange(minimum_threshold, 0.951, 0.01):
        selected = confidences >= threshold
        count = int(selected.sum())
        if not count:
            continue
        precision = float(accuracy_score(expected_values[selected], predictions[selected]))
        if precision >= target_precision:
            selected_threshold = round(float(threshold), 2)
            return selected_threshold, {
                "target_precision": target_precision,
                "minimum_threshold": minimum_threshold,
                "selected_count": count,
                "selected_coverage": round(count / len(expected), 6),
                "selected_precision": round(precision, 6),
            }
    return 0.95, {
        "target_precision": target_precision,
        "minimum_threshold": minimum_threshold,
        "selected_count": 0,
        "selected_coverage": 0.0,
        "selected_precision": 0.0,
    }


def _tune_base_model(
    prepared: PreparedDataset,
    random_seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    """Select regularization only within grouped training folds."""

    cross_validation = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_seed,
    )
    search = GridSearchCV(
        estimator=build_base_pipeline(random_seed),
        param_grid={"classifier__C": list(REGULARIZATION_GRID)},
        scoring={"macro_f1": "f1_macro", "accuracy": "accuracy", "neg_log_loss": "neg_log_loss"},
        refit="macro_f1",
        cv=cross_validation,
        n_jobs=1,
        return_train_score=True,
    )
    search.fit(prepared.train_texts, prepared.train_labels, groups=prepared.train_groups)
    candidates = []
    for index, parameters in enumerate(search.cv_results_["params"]):
        candidates.append(
            {
                "C": parameters["classifier__C"],
                "mean_train_macro_f1": round(
                    float(search.cv_results_["mean_train_macro_f1"][index]), 6
                ),
                "mean_validation_macro_f1": round(
                    float(search.cv_results_["mean_test_macro_f1"][index]), 6
                ),
                "std_validation_macro_f1": round(
                    float(search.cv_results_["std_test_macro_f1"][index]), 6
                ),
                "mean_validation_accuracy": round(
                    float(search.cv_results_["mean_test_accuracy"][index]), 6
                ),
                "mean_validation_log_loss": round(
                    float(-search.cv_results_["mean_test_neg_log_loss"][index]), 6
                ),
            }
        )
    selected = candidates[search.best_index_]
    return search.best_estimator_, {
        "strategy": "5-fold StratifiedGroupKFold by semantic template family",
        "selection_metric": "macro_f1",
        "selected_C": search.best_params_["classifier__C"],
        "selected_train_validation_gap": round(
            selected["mean_train_macro_f1"] - selected["mean_validation_macro_f1"], 6
        ),
        "candidates": candidates,
    }


def _evaluate_ood(
    model: CalibratedClassifierCV,
    texts: tuple[str, ...],
    selected_threshold: float,
) -> dict[str, Any]:
    """Measure whether ambiguous and unsupported cases are rejected safely."""

    probabilities = model.predict_proba(texts)
    confidences = probabilities.max(axis=1)
    predictions = model.predict(texts)
    auto_routed = confidences >= selected_threshold
    return {
        "count": len(texts),
        "selected_threshold": selected_threshold,
        "rejected_count": int((~auto_routed).sum()),
        "rejection_rate": round(float((~auto_routed).mean()), 6),
        "false_auto_route_count": int(auto_routed.sum()),
        "false_auto_route_rate": round(float(auto_routed.mean()), 6),
        "mean_confidence": round(float(confidences.mean()), 6),
        "maximum_confidence": round(float(confidences.max()), 6),
        "predicted_class_counts": dict(sorted(Counter(str(value) for value in predictions).items())),
    }


def train_and_persist(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    artifact_path: str | Path = DEFAULT_ARTIFACT_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    random_seed: int = 42,
    routing_threshold: float = 0.80,
) -> TrainingResult:
    """Tune, calibrate, evaluate untouched splits, and persist the model."""

    if not 0.0 <= routing_threshold <= 1.0:
        raise ValueError("routing_threshold must be between 0 and 1")
    resolved_dataset = Path(dataset_path)
    resolved_artifact = Path(artifact_path)
    resolved_metrics = Path(metrics_path)
    prepared = prepare_training_data(load_training_rows(resolved_dataset))

    base_pipeline, grouped_cv = _tune_base_model(prepared, random_seed)
    model = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_pipeline),
        method="sigmoid",
    )
    model.fit(prepared.calibration_texts, prepared.calibration_labels)
    label_order = tuple(str(label) for label in model.classes_)
    selected_threshold, threshold_selection = _select_routing_threshold(
        model,
        prepared.calibration_texts,
        prepared.calibration_labels,
        minimum_threshold=routing_threshold,
    )

    combined_texts = (*prepared.validation_texts, *prepared.challenge_texts)
    combined_labels = (*prepared.validation_labels, *prepared.challenge_labels)
    train_metrics = _evaluate(
        model, prepared.train_texts, prepared.train_labels, label_order, selected_threshold
    )
    calibration_metrics = _evaluate(
        model,
        prepared.calibration_texts,
        prepared.calibration_labels,
        label_order,
        selected_threshold,
    )
    standard_metrics = _evaluate(
        model,
        prepared.validation_texts,
        prepared.validation_labels,
        label_order,
        selected_threshold,
    )
    challenge_metrics = _evaluate(
        model,
        prepared.challenge_texts,
        prepared.challenge_labels,
        label_order,
        selected_threshold,
    )
    combined_metrics = _evaluate(
        model, combined_texts, combined_labels, label_order, selected_threshold
    )
    generalization_gap = {
        "accuracy": round(train_metrics["accuracy"] - combined_metrics["accuracy"], 6),
        "macro_f1": round(train_metrics["macro_f1"] - combined_metrics["macro_f1"], 6),
    }

    metadata: dict[str, Any] = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dataset_sha256": calculate_sha256(resolved_dataset),
        "random_seed": random_seed,
        "selected_routing_threshold": selected_threshold,
        "threshold_selection": threshold_selection,
        "label_order": list(label_order),
        "train_count": len(prepared.train_texts),
        "calibration_count": len(prepared.calibration_texts),
        "validation_count": len(prepared.validation_texts),
        "challenge_count": len(prepared.challenge_texts),
        "ood_count": len(prepared.ood_texts),
        "train_class_counts": dict(sorted(Counter(prepared.train_labels).items())),
        "calibration_class_counts": dict(sorted(Counter(prepared.calibration_labels).items())),
        "validation_class_counts": dict(sorted(Counter(prepared.validation_labels).items())),
        "challenge_class_counts": dict(sorted(Counter(prepared.challenge_labels).items())),
        "preprocessing": {
            "implementation": "EmailPreprocessor/PIIRedactor",
            "pii_free_model_text": True,
            "all_pairwise_split_text_overlap": 0,
            "all_pairwise_template_family_overlap": 0,
        },
        "estimator": {
            "features": "FeatureUnion(word TF-IDF 1-2 grams, character TF-IDF 3-5 grams)",
            "classifier": "L2 class-balanced LogisticRegression",
            "calibration": "sigmoid on dedicated group-disjoint calibration partition",
            "selected_C": grouped_cv["selected_C"],
        },
        "grouped_cross_validation": grouped_cv,
        "environment": {
            "python": platform.python_version(),
            "python_major_minor": ".".join(platform.python_version().split(".")[:2]),
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "train_metrics": train_metrics,
        "calibration_metrics": calibration_metrics,
        "standard_validation": standard_metrics,
        "challenge_validation": challenge_metrics,
        "combined_holdout": combined_metrics,
        "generalization_gap": generalization_gap,
        "ood_rejection": _evaluate_ood(model, prepared.ood_texts, selected_threshold),
        "validation": combined_metrics,
    }
    bundle = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "model": model,
        "base_pipeline": base_pipeline,
        "metadata": metadata,
    }
    resolved_artifact.parent.mkdir(parents=True, exist_ok=True)
    resolved_metrics.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, resolved_artifact, compress=3)
    metadata["artifact_sha256"] = calculate_sha256(resolved_artifact)
    resolved_metrics.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TrainingResult(resolved_artifact, resolved_metrics, metadata)

