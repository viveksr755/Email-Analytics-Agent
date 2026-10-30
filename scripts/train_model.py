"""Train, evaluate, and persist the genuine CTD classifier artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from email_analytics_agent.training import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_METRICS_PATH,
    train_and_persist,
)


def build_parser() -> argparse.ArgumentParser:
    """Create arguments for reproducible local model training."""

    parser = argparse.ArgumentParser(description="Train the Meridian CTD classifier")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--routing-threshold", type=float, default=0.80)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run training and print the persisted robustness summary as JSON."""

    args = build_parser().parse_args(argv)
    result = train_and_persist(
        dataset_path=args.dataset,
        artifact_path=args.artifact,
        metrics_path=args.metrics,
        random_seed=args.seed,
        routing_threshold=args.routing_threshold,
    )
    summary = {
        "artifact": str(result.artifact_path),
        "metrics": str(result.metrics_path),
        "model_version": result.metadata["model_version"],
        "train_count": result.metadata["train_count"],
        "calibration_count": result.metadata["calibration_count"],
        "validation_count": result.metadata["validation_count"],
        "challenge_count": result.metadata["challenge_count"],
        "selected_routing_threshold": result.metadata["selected_routing_threshold"],
        "selected_C": result.metadata["grouped_cross_validation"]["selected_C"],
        "standard_validation": result.metadata["standard_validation"],
        "challenge_validation": result.metadata["challenge_validation"],
        "combined_holdout": result.metadata["combined_holdout"],
        "generalization_gap": result.metadata["generalization_gap"],
        "ood_rejection": result.metadata["ood_rejection"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

