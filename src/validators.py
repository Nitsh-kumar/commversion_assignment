"""Deterministic input parsers and validators for budgets, postcodes, phones, and emails."""

from __future__ import annotations

import re

# Full UK postcode format (e.g. SW3 3RB, GU5 7AQ, EC1A 1BB)
_UK_POSTCODE_RE = re.compile(
    r"\b([A-Z]{1,2}\d[A-Z\d]?)\s*(\d[A-Z]{2})\b",
    re.IGNORECASE,
)

# Budget numbers with optional currency symbol, commas/decimals, and k/m/million/thousand suffix
_BUDGET_RE = re.compile(
    r"(?:£|gbp\s*)?(\d+(?:[.,]\d{3})*(?:[.,]\d+)?|\d+(?:\.\d+)?)\s*(million|mill|mn|m|thousand|k)?\b",
    re.IGNORECASE,
)

_EMAIL_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

# Extract candidate phone spans containing digits, spaces, +, -, parentheses
_PHONE_CANDIDATE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,18}\d)")


def parse_budget_gbp(text: str) -> int | None:
    """Parse natural-language GBP budget expressions into an integer amount in pounds.

    Supports formats such as '£1.5m', '1,200,000', '1.2 million', '950k', '£850,000'.
    Returns None if no plausible budget is found.
    """
    if not text or not text.strip():
        return None

    cleaned = text.strip().lower()
    # Ignore obvious postcodes or phone-like long strings without currency/unit hints
    for match in _BUDGET_RE.finditer(cleaned):
        raw_num, unit = match.group(1), (match.group(2) or "").lower()
        # Distinguish decimal dot (1.5m) from thousand-grouped commas (1,200,000)
        if "," in raw_num and "." not in raw_num:
            # Handle 1,200,000 or 1,5 (only treat comma as decimal if <=2 digits after single comma + unit)
            parts = raw_num.split(",")
            if len(parts) == 2 and len(parts[1]) <= 2 and unit:
                num_str = raw_num.replace(",", ".")
            else:
                num_str = raw_num.replace(",", "")
        else:
            num_str = raw_num.replace(",", "")

        try:
            value = float(num_str)
        except ValueError:
            continue

        if unit in {"m", "mn", "mill", "million"}:
            value *= 1_000_000
        elif unit in {"k", "thousand"}:
            value *= 1_000
        else:
            # Bare numbers below 1000 without k/m are usually bedroom counts or noise, not home budgets
            if value < 10_000:
                continue

        amount = int(round(value))
        if 10_000 <= amount <= 500_000_000:
            return amount

    return None


def normalize_postcode(text: str) -> str | None:
    """Extract and normalize a full UK postcode to uppercase 'OUTWARD INWARD' format.

    Example: 'gu5 7aq' -> 'GU5 7AQ', 'sw33rb' -> 'SW3 3RB'.
    """
    if not text:
        return None
    match = _UK_POSTCODE_RE.search(text.strip())
    if not match:
        return None
    outward = match.group(1).upper()
    inward = match.group(2).upper()
    return f"{outward} {inward}"


def validate_phone(text: str) -> str | None:
    """Validate a UK or international phone number and return normalized string, or None."""
    if not text:
        return None
    match = _PHONE_CANDIDATE_RE.search(text.strip())
    if not match:
        return None
    candidate = match.group(0).strip()
    digits = re.sub(r"\D", "", candidate)
    if candidate.startswith("+"):
        if 10 <= len(digits) <= 15:
            return "+" + digits
        return None
    # UK domestic numbers are typically 10 or 11 digits starting with 0
    if digits.startswith("0") and 10 <= len(digits) <= 11:
        return candidate
    if 10 <= len(digits) <= 15:
        return candidate
    return None


def validate_email(text: str) -> str | None:
    """Extract and validate an email address, returning lowercased address or None."""
    if not text:
        return None
    match = _EMAIL_RE.search(text.strip())
    if not match:
        return None
    return match.group(0).lower()


def format_gbp(amount: int) -> str:
    """Format an integer GBP amount nicely for user-facing messages."""
    return f"£{amount:,}"
