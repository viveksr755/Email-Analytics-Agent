"""Tests for leakage-resistant training, evaluation, and model persistence."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from email_analytics_agent.classification import ModelArtifactError, SklearnCTDClassifier
from email_analytics_agent.preprocessing import PIIRedactor
from email_analytics_agent.training import (
    DEFAULT_DATASET_PATH,
    ARTIFACT_SCHEMA_VERSION,
    EXPECTED_LABELS,
    load_training_rows,
    prepare_training_data,
    train_and_persist,
)


def test_training_corpus_is_balanced_pii_safe_and_leakage_free() -> None:
    """Verify the explicit partitions remain valid after production redaction."""

    rows = load_training_rows()
    prepared = prepare_training_data(rows)

    assert len(rows) == 692
    assert Counter(prepared.train_labels) == Counter({label: 120 for label in EXPECTED_LABELS})
    assert Counter(prepared.calibration_labels) == Counter({label: 20 for label in EXPECTED_LABELS})
    assert Counter(prepared.validation_labels) == Counter({label: 20 for label in EXPECTED_LABELS})
    assert Counter(prepared.challenge_labels) == Counter({label: 10 for label in EXPECTED_LABELS})
    assert len(prepared.ood_texts) == 12
    all_splits = (
        prepared.train_texts,
        prepared.calibration_texts,
        prepared.validation_texts,
        prepared.challenge_texts,
        prepared.ood_texts,
    )
    assert all(not (set(left) & set(right)) for index, left in enumerate(all_splits) for right in all_splits[index + 1 :])
    family_sets = tuple(prepared.families_by_split.values())
    assert all(not (left & right) for index, left in enumerate(family_sets) for right in family_sets[index + 1 :])
    assert all(not PIIRedactor().contains_pii(text) for split in all_splits for text in split)
    assert all("[ACCOUNT_NUMBER]" in text for text in prepared.train_texts)


def test_default_artifact_contains_real_trained_sklearn_components() -> None:
    """Prove runtime serves TF-IDF plus calibrated logistic regression."""

    classifier = SklearnCTDClassifier()
    features = classifier.pipeline.named_steps["features"]
    base_classifier = classifier.pipeline.named_steps["classifier"]
    feature_map = dict(features.transformer_list)

    assert isinstance(features, FeatureUnion)
    assert isinstance(feature_map["word"], TfidfVectorizer)
    assert isinstance(feature_map["character"], TfidfVectorizer)
    assert isinstance(base_classifier, LogisticRegression)
    assert isinstance(classifier.model, CalibratedClassifierCV)
    assert set(classifier.model.classes_) == set(EXPECTED_LABELS)
    assert classifier.metadata["validation"]["accuracy"] >= 0.90
    assert classifier.metadata["validation"]["macro_f1"] >= 0.90
    assert classifier.metadata["validation"]["accuracy"] < 1.0
    assert classifier.metadata["challenge_validation"]["accuracy"] < classifier.metadata["standard_validation"]["accuracy"]
    assert classifier.metadata["grouped_cross_validation"]["selected_C"] <= 1.0
    assert classifier.metadata["ood_rejection"]["rejection_rate"] >= 0.80
    assert classifier.metadata["generalization_gap"]["macro_f1"] > 0.0
    selected = classifier.metadata["validation"]["routing_thresholds"]["0.80"]
    assert selected["coverage"] >= 0.70
    assert selected["selective_accuracy"] >= 0.95
    assert classifier.metadata["challenge_validation"]["accuracy"] >= 0.70
    assert classifier.metadata["ood_rejection"]["false_auto_route_rate"] <= 0.10


def test_prediction_confidence_is_exact_predict_proba_maximum() -> None:
    """Verify orchestration confidence comes from the fitted estimator itself."""

    classifier = SklearnCTDClassifier()
    text = "Subject: Wrong fee\nBody: I am raising a complaint about an incorrect charge."
    distribution = classifier.probability_distribution(text)
    prediction = classifier.predict(text)

    assert np.isclose(sum(distribution.values()), 1.0)
    assert prediction.label.department == max(distribution, key=distribution.get)
    assert np.isclose(prediction.confidence, max(distribution.values()), atol=1e-6)


def test_training_is_reproducible_and_artifact_round_trips(tmp_path: Path) -> None:
    """Verify a fixed seed yields identical validation behavior and predictions."""

    first = train_and_persist(
        DEFAULT_DATASET_PATH,
        tmp_path / "first.joblib",
        tmp_path / "first.metrics.json",
        random_seed=42,
    )
    second = train_and_persist(
        DEFAULT_DATASET_PATH,
        tmp_path / "second.joblib",
        tmp_path / "second.metrics.json",
        random_seed=42,
    )
    first_classifier = SklearnCTDClassifier(first.artifact_path, first.metrics_path)
    second_classifier = SklearnCTDClassifier(second.artifact_path, second.metrics_path)
    text = "Subject: Currency conversion\nBody: Quote the treasury exchange rate for our FX trade."

    assert first.metadata["validation"] == second.metadata["validation"]
    assert first_classifier.probability_distribution(text) == second_classifier.probability_distribution(text)


def test_missing_or_incompatible_artifacts_fail_closed(tmp_path: Path) -> None:
    """Ensure runtime never silently replaces a missing model with heuristics."""

    with pytest.raises(ModelArtifactError, match="run:"):
        SklearnCTDClassifier(tmp_path / "missing.joblib", tmp_path / "missing.json")

    metrics = json.loads(
        (Path(__file__).parents[1] / "artifacts" / "ctd_classifier.metrics.json").read_text(
            encoding="utf-8"
        )
    )
    metrics["artifact_schema_version"] = "999"
    incompatible = tmp_path / "incompatible.json"
    incompatible.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="schema"):
        SklearnCTDClassifier(
            Path(__file__).parents[1] / "artifacts" / "ctd_classifier.joblib",
            incompatible,
        )

    metrics["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
    metrics["environment"]["scikit_learn"] = "0.0.0"
    wrong_environment = tmp_path / "wrong-environment.json"
    wrong_environment.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="runtime differs"):
        SklearnCTDClassifier(
            Path(__file__).parents[1] / "artifacts" / "ctd_classifier.joblib",
            wrong_environment,
        )

