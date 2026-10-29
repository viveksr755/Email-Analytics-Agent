"""Canonical CTD taxonomy and queue mapping for the proof of concept."""

from __future__ import annotations

from .models import CTDLabel


CTD_BY_DEPARTMENT: dict[str, CTDLabel] = {
    "Loans": CTDLabel("Lending", "Loan servicing or application", "Loans"),
    "Credit": CTDLabel("Credit products", "Card or credit request", "Credit"),
    "Complaints": CTDLabel("Customer service", "Complaint or grievance", "Complaints"),
    "Treasury": CTDLabel("Markets", "Treasury or foreign exchange instruction", "Treasury"),
}

UNKNOWN_CTD = CTDLabel("Unclassified", "Manual review required", "RM Review")

QUEUE_BY_DEPARTMENT: dict[str, str] = {
    "Loans": "loans-operations",
    "Credit": "credit-operations",
    "Complaints": "complaints-resolution",
    "Treasury": "treasury-operations",
    "RM Review": "rm-review",
}


def queue_for(label: CTDLabel) -> str:
    """Map a CTD label to an operational queue, defaulting safely to RM review."""

    return QUEUE_BY_DEPARTMENT.get(label.department, "rm-review")

