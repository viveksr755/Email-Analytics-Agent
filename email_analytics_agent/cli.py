"""Command-line demonstration of the local polling-shaped analytics flow."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from .factory import build_default_system
from .models import EmailMessage


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser with repository-friendly defaults."""

    parser = argparse.ArgumentParser(description="Run the Meridian Email Analytics Agent POC")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/mock_emails.json"),
        help="JSON array containing synthetic email records",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("logs/audit.jsonl"),
        help="append-only JSONL audit destination",
    )
    return parser


def load_messages(path: Path) -> list[EmailMessage]:
    """Load a JSON array and convert each record into a typed email."""

    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("input JSON must contain an array of email objects")
    return [EmailMessage.from_dict(value) for value in values]


def run(input_path: Path, audit_path: Path) -> dict[str, object]:
    """Process Graph-shaped fixture emails and summarize one deterministic run.

    The JSON fixture represents messages immediately after Graph conversion.
    Live Graph polling is exposed through ``build_graph_polling_system`` and is
    kept out of this offline demonstration so tests never require credentials.
    """

    messages = load_messages(input_path)
    ingested = len(messages)
    system = build_default_system(audit_path)
    decisions: Counter[str] = Counter()
    queues: Counter[str] = Counter()

    for message in messages:
        result = system.manager.process_email(message)
        decisions[result.decision] += 1
        queues[result.target_queue] += 1

    return {
        "ingested": ingested,
        "stored": system.document_store.count(),
        "decisions": dict(sorted(decisions.items())),
        "queues": dict(sorted(queues.items())),
        "llm_calls": len(system.llm_client.calls),
        "audit_entries": len(system.audit_log.entries()),
        "audit_integrity": system.audit_log.verify_integrity(),
        "audit_path": str(audit_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the POC and print a machine-readable run summary."""

    args = build_parser().parse_args(argv)
    print(json.dumps(run(args.input, args.audit), indent=2, sort_keys=True))
    return 0

