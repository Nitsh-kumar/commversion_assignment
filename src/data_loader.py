"""CSV loaders for Ashcombe & Rowe Estates data sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.validators import normalize_postcode


@dataclass(frozen=True)
class FaqEntry:
    """Single FAQ row from faq.csv."""

    faq_id: str
    category: str
    question: str
    answer: str


@dataclass(frozen=True)
class PropertyRecord:
    """Single property listing from properties_for_buyers.csv."""

    property_id: str
    property_name: str
    address: str
    town: str
    postcode: str
    listing_category: str  # new_build | resale
    property_type: str
    bedrooms: int
    bathrooms: int
    price_gbp: int
    status: str  # available | under_offer | sold
    description: str
    branch: str
    listed_date: str


def _resolve_csv(data_dir: Path, primary: str, fallback: str) -> Path:
    """Resolve canonical CSV name with fallback to original workspace filename."""
    p = data_dir / primary
    if p.exists():
        return p
    alt = data_dir.parent / fallback
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find {primary} in {data_dir} or {fallback} in {data_dir.parent}")


def load_faqs(data_dir: Path) -> list[FaqEntry]:
    """Load FAQ Q&A entries from faq.csv."""
    path = _resolve_csv(data_dir, "faq.csv", "faq 1.csv")
    df = pd.read_csv(path, dtype=str).fillna("")
    return [
        FaqEntry(
            faq_id=row["faq_id"].strip(),
            category=row["category"].strip(),
            question=row["question"].strip(),
            answer=row["answer"].strip(),
        )
        for _, row in df.iterrows()
    ]


def load_properties(data_dir: Path) -> list[PropertyRecord]:
    """Load property listings from properties_for_buyers.csv."""
    path = _resolve_csv(data_dir, "properties_for_buyers.csv", "properties_for_buyers 1.csv")
    df = pd.read_csv(path, dtype=str).fillna("")
    records: list[PropertyRecord] = []
    for _, row in df.iterrows():
        norm_pc = normalize_postcode(row["postcode"]) or row["postcode"].strip().upper()
        records.append(
            PropertyRecord(
                property_id=row["property_id"].strip(),
                property_name=row["property_name"].strip(),
                address=row["address"].strip(),
                town=row["town"].strip(),
                postcode=norm_pc,
                listing_category=row["listing_category"].strip().lower(),
                property_type=row["property_type"].strip(),
                bedrooms=int(row["bedrooms"]),
                bathrooms=int(row["bathrooms"]),
                price_gbp=int(row["price_gbp"]),
                status=row["status"].strip().lower(),
                description=row["description"].strip(),
                branch=row["branch"].strip(),
                listed_date=row["listed_date"].strip(),
            )
        )
    return records


def load_seller_coverage(data_dir: Path) -> set[str]:
    """Load normalized UK postcodes covered for selling from seller_coverage.csv."""
    path = _resolve_csv(data_dir, "seller_coverage.csv", "seller_coverage 1.csv")
    df = pd.read_csv(path, dtype=str).fillna("")
    postcodes: set[str] = set()
    for raw in df["postcode"]:
        norm = normalize_postcode(raw)
        if norm:
            postcodes.add(norm)
    return postcodes
