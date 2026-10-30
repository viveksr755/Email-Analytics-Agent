"""Generate a leakage-resistant synthetic corpus for CTD model training."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


TRAIN_TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {
    "Loans": {
        "subjects": (
            "Help with housing finance application",
            "Mortgage approval status",
            "EMI repayment schedule request",
            "Loan disbursement follow up",
            "Personal borrowing enquiry",
            "Early loan closure quote",
            "Education loan document check",
            "Vehicle finance installment issue",
            "Card matter closed - property funding needed",
            "Complaint resolved - borrowing support required",
        ),
        "bodies": (
            "Please explain the documents needed before my home loan can be approved.",
            "The mortgage application is complete but the sanction letter has not arrived.",
            "I need a revised EMI schedule for my personal loan repayment.",
            "Please confirm when the approved loan amount will be disbursed.",
            "Send the foreclosure amount and charges for closing this loan early.",
            "Can the tenure of my housing loan be extended to reduce installments?",
            "The education loan application needs an update before college admission.",
            "My vehicle finance EMI was debited twice and needs servicing support.",
            "I want to understand the interest rate applied to the mortgage balance.",
            "Please provide the outstanding loan statement and repayment history.",
            "This is not a card-limit or FX request; I need funds secured against my home.",
            "The transfer complaint is closed; please provide the borrowing payoff figure.",
        ),
    },
    "Credit": {
        "subjects": (
            "Credit card spending limit change",
            "Request for card upgrade",
            "Credit line review",
            "Replacement card servicing",
            "Corporate card activation",
            "Card payment due date question",
            "Supplementary credit card request",
            "International card usage enablement",
            "Mortgage completed - card servicing now required",
            "Earlier grievance closed - spending facility request",
        ),
        "bodies": (
            "Please increase the spending limit available on my credit card.",
            "I qualify for an upgrade and want a premium card with travel benefits.",
            "Review my credit line because the current card limit is insufficient.",
            "The replacement credit card has arrived and needs to be activated.",
            "Please enable overseas transactions for my card before the business trip.",
            "I need the due date and minimum payment for this card statement.",
            "Issue a supplementary credit card for a family member.",
            "Explain the annual fee and reward points on the upgraded card.",
            "My contactless card function is disabled and requires servicing.",
            "Please convert this large card purchase into monthly installments.",
            "This is not a loan or FX trade; please raise the spending cap on my card.",
            "The old complaint is closed; activate the replacement plastic and set its limit.",
        ),
    },
    "Complaints": {
        "subjects": (
            "Formal complaint about failed transfer",
            "Incorrect banking fee grievance",
            "Poor branch service complaint",
            "Disputed debit needs investigation",
            "Complaint about delayed reversal",
            "Escalation for unresolved service issue",
            "ATM cash withdrawal grievance",
            "Incorrect account charge complaint",
            "Loan servicing is not requested - formal grievance",
            "Treasury deal completed - unresolved service failure",
        ),
        "bodies": (
            "I am raising a complaint because the transfer failed but funds were debited.",
            "The bank charged an incorrect fee and I want the grievance investigated.",
            "Service at the branch was unacceptable and this complaint needs resolution.",
            "A disputed debit remains on my statement after multiple support calls.",
            "The promised reversal is delayed, so please escalate my complaint.",
            "My earlier service issue remains unresolved and I am dissatisfied.",
            "The ATM dispensed no cash although the account was charged.",
            "Please investigate and refund the incorrect maintenance charge.",
            "I received no response to the grievance submitted last week.",
            "The failed payment caused a loss and the complaint requires urgent review.",
            "Do not process a loan change; investigate the duplicated EMI debit as a complaint.",
            "The FX deal itself is correct, but the missing reversal is an unresolved grievance.",
        ),
    },
    "Treasury": {
        "subjects": (
            "Foreign exchange trade instruction",
            "Currency forward booking",
            "Treasury settlement confirmation",
            "FX hedge requirement",
            "Spot currency deal request",
            "Corporate remittance rate quote",
            "Treasury trade value date change",
            "Foreign currency settlement account",
            "Card payment irrelevant - currency hedge instruction",
            "Loan request closed - new foreign exchange deal",
        ),
        "bodies": (
            "Please book a USD INR foreign exchange trade at the agreed treasury rate.",
            "We require a currency forward contract to hedge the upcoming import payment.",
            "Confirm settlement details and value date for yesterday's FX transaction.",
            "Provide a treasury quote for hedging our foreign currency exposure.",
            "Execute the spot deal and send the trade confirmation to our finance team.",
            "Please quote the exchange rate for this corporate outward remittance.",
            "Change the value date on the treasury transaction before settlement.",
            "Use the nominated foreign currency account for trade settlement.",
            "We need an FX contract for the exporter receivable due next month.",
            "Send the deal ticket for the completed currency hedge transaction.",
            "This is not a credit-card request; lock the exchange price for our import exposure.",
            "The borrowing case is closed; execute the new spot currency conversion instruction.",
        ),
    },
}


VALIDATION_TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {
    "Loans": {
        "subjects": (
            "Funding needed to purchase a flat",
            "Borrowing payoff details requested",
            "College finance application question",
            "Car loan approval delayed",
        ),
        "bodies": (
            "What paperwork remains before the mortgage for my new flat is sanctioned?",
            "Calculate the amount required to repay and close my borrowing facility.",
            "The student loan must be released before the university payment deadline.",
            "Please correct the installment schedule on my vehicle loan account.",
            "I need a statement showing principal and interest paid on this home finance.",
        ),
    },
    "Credit": {
        "subjects": (
            "Need more purchasing capacity on card",
            "Premium plastic eligibility question",
            "New add-on card for spouse",
            "Enable card for overseas trip",
        ),
        "bodies": (
            "Can the available credit limit on this card be raised before a large purchase?",
            "Tell me whether this credit card can be upgraded to the premium tier.",
            "I want an additional card issued under the same credit account.",
            "Allow international merchant payments on my card while I travel.",
            "Please explain the card statement balance, due amount, and reward points.",
        ),
    },
    "Complaints": {
        "subjects": (
            "Unresolved debit service grievance",
            "Dissatisfied with transfer handling",
            "Cash machine dispute escalation",
            "Wrong charge requires resolution",
        ),
        "bodies": (
            "I want to complain because money was deducted for an unsuccessful payment.",
            "Your support team has not resolved my grievance about the delayed refund.",
            "The cash machine gave nothing, yet the withdrawal appears on my statement.",
            "Reverse this wrongly applied service charge and investigate the error.",
            "I am dissatisfied with repeated delays and need the issue formally escalated.",
        ),
    },
    "Treasury": {
        "subjects": (
            "Protect upcoming dollar payment",
            "Exchange deal confirmation needed",
            "Corporate currency conversion quote",
            "Value date correction for FX deal",
        ),
        "bodies": (
            "Arrange a forward contract so the importer is protected from currency movement.",
            "Send confirmation of the foreign exchange deal and its settlement instructions.",
            "Quote the treasury rate for converting the corporate dollar balance into rupees.",
            "The settlement date on our FX trade needs to move to the next business day.",
            "Please execute the currency hedge and provide the completed deal ticket.",
        ),
    },
}


CALIBRATION_TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {
    "Loans": {
        "subjects": (
            "Property finance processing update",
            "Borrowing repayment assistance",
            "Installment tenure amendment",
            "Loan settlement figure",
        ),
        "bodies": (
            "Confirm the remaining steps before the housing loan is approved.",
            "Provide a fresh repayment table for the borrowing account.",
            "I want to shorten the mortgage tenure and revise the EMI.",
            "Send the final amount needed to settle the vehicle loan.",
            "When will the sanctioned education finance be released?",
        ),
    },
    "Credit": {
        "subjects": (
            "Card facility amendment",
            "Spending ceiling review",
            "Additional card request",
            "Credit plastic activation",
        ),
        "bodies": (
            "Increase the purchase limit available on this credit card.",
            "Activate the newly delivered replacement card.",
            "Issue an additional card under my existing credit facility.",
            "Enable overseas usage before my trip next week.",
            "Explain the reward balance and payment due on the card statement.",
        ),
    },
    "Complaints": {
        "subjects": (
            "Service failure escalation",
            "Debit error grievance",
            "Refund delay complaint",
            "Cash withdrawal dispute",
        ),
        "bodies": (
            "Escalate my complaint because the failed payment was still debited.",
            "The incorrect charge has not been refunded despite earlier calls.",
            "I am dissatisfied with the branch response to my grievance.",
            "No cash was dispensed, but the ATM withdrawal remains posted.",
            "Investigate the delayed reversal and provide a complaint reference.",
        ),
    },
    "Treasury": {
        "subjects": (
            "Currency risk management instruction",
            "Foreign exchange settlement update",
            "Corporate conversion rate request",
            "FX deal date amendment",
        ),
        "bodies": (
            "Book a forward contract for the upcoming foreign currency payment.",
            "Confirm the value date and settlement account for the FX trade.",
            "Quote the treasury rate for converting dollars into rupees.",
            "Execute a spot currency transaction for the corporate account.",
            "Provide the confirmation ticket for our completed hedge.",
        ),
    },
}


CHALLENGE_EXAMPLES: dict[str, tuple[tuple[str, str], ...]] = {
    "Loans": (
        ("Not a card query", "The credit card is fine; I need the payoff amount for my housing finance."),
        ("Complaint withdrawn", "This is not a complaint about service; send the mortgage foreclosure quote."),
        ("Property funding", "The price is in USD, but this request is only for home finance approval."),
        ("plz help", "hom lon emi schdule pls"),
        ("New request", "Previous subject: card limit. Actual request: release education borrowing funds."),
        ("No dispute", "The annual fee and FX rate are fine; calculate the remaining car finance principal."),
        ("Sanction query", "A treasury quote is attached as income proof; when will property funding be approved?"),
        ("Tenure help", "The card works and my branch complaint is closed; extend the education borrowing tenure."),
        ("Changed requirement", "Quoted: increase card limit. New request: settlement figure for vehicle finance."),
        ("closure pls", "need brwing clsure amnt for flat fnance"),
    ),
    "Credit": (
        ("Mortgage completed", "The mortgage is paid; please raise my credit card spending cap."),
        ("Complaint closed", "My earlier complaint is closed; activate the replacement card."),
        ("Trip enablement", "Travelling for an FX conference; enable international merchant use on my plastic."),
        ("plz help", "crdit crd limt increse"),
        ("Actual request", "RE loan documents. Actual request: supplementary card for my spouse."),
        ("No grievance", "The wrong fee was refunded; now issue the premium card I requested."),
        ("Business travel", "Treasury booked the FX; I only need the corporate card activated."),
        ("Loan paid", "The personal loan is closed; provide an add-on plastic for my spouse."),
        ("Thread moved", "Quoted: mortgage statement. Current request: enable contactless card payments."),
        ("cap pls", "increase purchsing cap on my plstic"),
    ),
    "Complaints": (
        ("Duplicate EMI debit", "Do not process a loan request; I am complaining about a duplicate EMI debit."),
        ("FX issue resolved", "The FX trade was fine; my complaint is that the branch ignored the reversal."),
        ("Annual fee error", "My credit card limit is okay; the wrong annual fee remains unresolved."),
        ("urgent", "complnt: atm cash no dispnsed but debtd"),
        ("Missing funds", "Treasury said settled; funds are missing and I need a grievance escalated."),
        ("Card rejection grievance", "I do not want a new card; complain about repeated unexplained declines."),
        ("Loan delay escalation", "Keep the mortgage terms unchanged and escalate the service delay as a grievance."),
        ("Bad treasury service", "The requested FX rate was ignored; log a formal complaint about the handling."),
        ("Different issue", "Quoted: vehicle loan schedule. Current issue: branch staff refused my refund complaint."),
        ("wrong debit", "not askng for credit; fix unexplned debit and escalte issue"),
    ),
    "Treasury": (
        ("Exposure hedge", "This is not a loan; book a currency exposure hedge."),
        ("Import conversion", "The card payment is an import expense; quote USD INR conversion."),
        ("Amend value date", "The complaint is resolved; now amend the FX value date."),
        ("urgent", "plz bok usd inr frwrd"),
        ("New instruction", "Previous loan request closed. Execute a spot currency deal."),
        ("Export receipt", "A loan document is attached, but hedge the foreign receivable due next month."),
        ("Corporate conversion", "This is not a card-limit change; convert the company's euro balance."),
        ("Extend contract", "Do not open a credit facility; amend the maturity of our currency forward."),
        ("Thread changed", "Quoted complaint below. Current instruction: settle the foreign exchange trade."),
        ("deal pls", "need fx convrsn qte and val date cnfrmtn"),
    ),
}


OOD_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("General help", "Please help with my account."),
    ("Two requests", "I need both loan closure and a credit card limit increase."),
    ("Mixed case", "I am complaining about loan EMI and also need a repayment schedule."),
    ("Funding choice", "Should we hedge with a forward or take a working-capital loan?"),
    ("KYC update", "Please update KYC and proof of address."),
    ("Profile servicing", "Please change the registered mobile number."),
    ("Unclear", "Need assistance with banking."),
    ("Products", "Can someone explain loans, cards and exchange services?"),
    ("Thread summary", "The email chain mentions a complaint, a mortgage and an FX conversion."),
    ("Ambiguous debit", "The payment relates to my card and loan but I do not know what action is needed."),
    ("Service options", "Should this go to credit, treasury or the complaints team?"),
    ("Call back", "Please ask a relationship manager to call me about several products."),
)


def build_training_dataset() -> list[dict[str, object]]:
    """Create grouped train/calibration/holdout/challenge/OOD partitions."""

    rows: list[dict[str, object]] = []
    sequence = 1
    received = datetime(2026, 2, 1, 9, tzinfo=timezone.utc)

    def append_example(
        subject: str,
        body: str,
        department: str,
        split: str,
        template_family: str,
    ) -> None:
        """Append one synthetic row with non-label-bearing PII and metadata."""

        nonlocal sequence
        account = f"88007766{sequence:04d}"
        customer = f"CUST{sequence:06d}"
        email = f"synthetic{sequence}@example.test"
        rows.append(
            {
                "message_id": f"TRAIN-{sequence:04d}",
                "sender": email,
                "subject": subject,
                "body": (
                    f"{body} Secured identifiers: account {account}, "
                    f"customer ID {customer}; contact {email}."
                ),
                "received_at": (received + timedelta(minutes=sequence)).isoformat().replace(
                    "+00:00", "Z"
                ),
                "expected_department": department,
                "split": split,
                "template_family": template_family,
            }
        )
        sequence += 1

    for split, family_name, templates_by_department in (
        ("train", "core", TRAIN_TEMPLATES),
        ("calibration", "calibration", CALIBRATION_TEMPLATES),
        ("validation", "holdout", VALIDATION_TEMPLATES),
    ):
        for department, templates in templates_by_department.items():
            for subject_index, subject in enumerate(templates["subjects"]):
                for body in templates["bodies"]:
                    append_example(
                        subject,
                        body,
                        department,
                        split,
                        f"{department.lower()}-{family_name}-subject-{subject_index + 1}",
                    )

    for department, examples in CHALLENGE_EXAMPLES.items():
        for example_index, (subject, body) in enumerate(examples, start=1):
            append_example(
                subject,
                body,
                department,
                "challenge",
                f"{department.lower()}-challenge-{example_index}",
            )

    for example_index, (subject, body) in enumerate(OOD_EXAMPLES, start=1):
        append_example(subject, body, "RM Review", "ood", f"ambiguous-ood-{example_index}")
    return rows


def main() -> None:
    """Write the deterministic training corpus to the repository data directory."""

    destination = Path(__file__).resolve().parents[1] / "data" / "training_emails.json"
    rows = build_training_dataset()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "calibration", "validation", "challenge", "ood")
    }
    print(f"Wrote {split_counts} to {destination}")


if __name__ == "__main__":
    main()

