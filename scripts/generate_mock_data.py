"""Regenerate the deterministic 20-message synthetic banking dataset."""

from __future__ import annotations

import json
from pathlib import Path


DEPARTMENTS = {
    "Loans": (
        "Home loan application {number}",
        "Please review my home loan mortgage application. My EMI account is {account} and customer ID is {customer}.",
    ),
    "Credit": (
        "Credit card limit request {number}",
        "Please increase the limit on my credit card. The linked account is {account} and my PAN is {pan}.",
    ),
    "Complaints": (
        "Complaint about failed transfer {number}",
        "I want to raise a complaint for a failed transfer and incorrect fee. Call me on {phone}; customer ID {customer}.",
    ),
    "Treasury": (
        "Treasury FX instruction {number}",
        "Please book an FX forward treasury trade and confirm the settlement account {account}. Contact {email}.",
    ),
}


def build_dataset() -> list[dict[str, str]]:
    """Build five examples per department with synthetic, non-real PII."""

    rows: list[dict[str, str]] = []
    number = 1
    for department, (subject, body) in DEPARTMENTS.items():
        for variant in range(1, 6):
            rows.append(
                {
                    "message_id": f"MSG-{number:03d}",
                    "sender": f"synthetic.customer{number}@example.test",
                    "subject": subject.format(number=variant),
                    "body": body.format(
                        account=f"99110022{number:04d}",
                        customer=f"CUST{number:06d}",
                        pan=f"ABCDE{number:04d}F",
                        phone=f"+91 90000 {number:05d}",
                        email=f"treasury{number}@example.test",
                    ),
                    "received_at": f"2026-01-{number:02d}T09:00:00Z",
                    "expected_department": department,
                }
            )
            number += 1
    return rows


def main() -> None:
    """Write the synthetic dataset to the repository data directory."""

    destination = Path(__file__).resolve().parents[1] / "data" / "mock_emails.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(build_dataset(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(build_dataset())} messages to {destination}")


if __name__ == "__main__":
    main()

