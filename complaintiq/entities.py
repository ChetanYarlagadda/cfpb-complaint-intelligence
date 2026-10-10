"""spaCy entity extraction (ORG, MONEY, DATE) with automatic spot checks against known structure.

CFPB narratives are scrubbed before publication: names and most dates become XXXX or
XX/XX/XXXX, while dollar amounts are kept and wrapped in braces, e.g. {$200.00}. That gives
two cheap references for a spot check: the brace amounts for MONEY, and the complaint's
company field (matched through a small alias list) for ORG.
"""

from __future__ import annotations

import re

import pandas as pd

BRACE_AMOUNT = re.compile(r"\{\$([\d,]+(?:\.\d{1,2})?)\}")
REDACTED_DATE = re.compile(r"X{2}/X{2}/(?:X{4}|\d{4})|X{2}/X{2}")
REDACTION_ONLY = re.compile(r"^[X\s/]+$")

# Brand names consumers use for the companies named in the complaint's `company` field.
COMPANY_ALIASES: dict[str, list[str]] = {
    "Block, Inc.": ["cash app", "cashapp", "block, inc", "block inc"],
    "Early Warning Services, LLC": ["zelle", "early warning"],
    "Paypal Holdings, Inc": ["paypal", "venmo", "xoom"],
    "Chime Financial Inc": ["chime"],
    "JPMORGAN CHASE & CO.": ["chase", "jpmorgan", "jp morgan"],
    "WELLS FARGO & COMPANY": ["wells fargo"],
    "BANK OF AMERICA, NATIONAL ASSOCIATION": ["bank of america", "bofa"],
    "Coinbase, Inc.": ["coinbase"],
    "ROBINHOOD MARKETS INC.": ["robinhood"],
    "WESTERN UNION COMPANY, THE": ["western union"],
    "TransferWise Ltd": ["transferwise", "wise.com"],
    "CAPITAL ONE FINANCIAL CORPORATION": ["capital one"],
    "CITIBANK, N.A.": ["citibank", "citi"],
    "Winklevoss Exchange LLC": ["gemini"],
    "NAVY FEDERAL CREDIT UNION": ["navy federal", "nfcu"],
    "Remitly, Inc.": ["remitly"],
    "MONEYGRAM PAYMENT SYSTEMS WORLDWIDE INC": ["moneygram", "money gram"],
}


def brace_amounts(narrative: str) -> list[float]:
    """Dollar amounts the CFPB wrapped in braces, e.g. {$1,200.00} -> 1200.0."""
    return [float(m.replace(",", "")) for m in BRACE_AMOUNT.findall(narrative)]


def _amount(text: str) -> float | None:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", text)
    return float(m.group(0).replace(",", "")) if m else None


def mentions_alias(text: str, aliases: list[str]) -> bool:
    """Whole-word, case-insensitive match of any alias."""
    low = text.lower()
    return any(re.search(rf"(?<![a-z]){re.escape(a)}(?![a-z])", low) for a in aliases)


def extract(nlp, narratives: list[str]) -> list[list[tuple[str, str]]]:
    """(label, text) of every ORG, MONEY and DATE entity per narrative."""
    keep = {"ORG", "MONEY", "DATE"}
    return [[(e.label_, e.text) for e in doc.ents if e.label_ in keep] for doc in nlp.pipe(narratives, batch_size=64)]


def spot_check(df: pd.DataFrame, ents: list[list[tuple[str, str]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-document comparison with the brace amounts and the company field; plus a summary table."""
    rows = []
    for rec, doc_ents in zip(df.itertuples(), ents):
        orgs = [t for lab, t in doc_ents if lab == "ORG"]
        money = [t for lab, t in doc_ents if lab == "MONEY"]
        dates = [t for lab, t in doc_ents if lab == "DATE"]
        ref_amounts = set(brace_amounts(rec.narrative))
        found_amounts = {a for a in map(_amount, money) if a is not None}
        aliases = COMPANY_ALIASES.get(rec.company)
        rows.append(
            {
                "complaint_id": rec.complaint_id,
                "company": rec.company,
                "n_org": len(orgs),
                "n_org_redaction_only": sum(bool(REDACTION_ONLY.match(o)) for o in orgs),
                "n_money": len(money),
                "n_date": len(dates),
                "n_date_redacted": sum(bool(REDACTED_DATE.search(d)) for d in dates),
                "has_redacted_date_in_text": bool(REDACTED_DATE.search(rec.narrative)),
                "n_brace_amounts": len(ref_amounts),
                "money_amounts_found": len(ref_amounts & found_amounts),
                "money_spurious": len(found_amounts - ref_amounts),
                "company_known": aliases is not None,
                "company_in_text": aliases is not None and mentions_alias(rec.narrative, aliases),
                "company_in_orgs": aliases is not None and any(mentions_alias(o, aliases) for o in orgs),
            }
        )
    per_doc = pd.DataFrame(rows)
    known = per_doc[per_doc.company_in_text]
    summary = pd.DataFrame(
        [
            {"check": "documents checked", "value": len(per_doc)},
            {"check": "ORG entities per document (mean)", "value": per_doc.n_org.mean()},
            {"check": "ORG: share of ORG entities that are only a redaction mark (XXXX)", "value": per_doc.n_org_redaction_only.sum() / max(1, per_doc.n_org.sum())},
            {"check": "MONEY: share of brace amounts recovered", "value": per_doc.money_amounts_found.sum() / max(1, per_doc.n_brace_amounts.sum())},
            {"check": "MONEY: share of extracted amounts that match a brace amount", "value": per_doc.money_amounts_found.sum() / max(1, per_doc.money_amounts_found.sum() + per_doc.money_spurious.sum())},
            {"check": "DATE: share of documents whose text contains a redacted date (XX/XX/XXXX)", "value": per_doc.has_redacted_date_in_text.mean()},
            {"check": "DATE: DATE entities per document (mean)", "value": per_doc.n_date.mean()},
            {"check": "DATE: share of DATE entities that are redacted dates (XX/XX/XXXX)", "value": per_doc.n_date_redacted.sum() / max(1, per_doc.n_date.sum())},
            {"check": "documents naming their company (alias list)", "value": len(known)},
            {"check": "ORG: share of those where an ORG entity names the company", "value": known.company_in_orgs.mean() if len(known) else float("nan")},
        ]
    )
    return per_doc, summary
