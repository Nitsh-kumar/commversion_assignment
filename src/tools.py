"""Data tools: deterministic property lookup, seller coverage check, and FAISS FAQ RAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.data_loader import (
    FaqEntry,
    PropertyRecord,
    load_faqs,
    load_properties,
    load_seller_coverage,
)
from src.validators import format_gbp, normalize_postcode

_STOPWORDS = {
    "the",
    "a",
    "an",
    "in",
    "on",
    "at",
    "of",
    "for",
    "property",
    "properties",
    "home",
    "house",
    "flat",
    "apartment",
    "townhouse",
    "penthouse",
    "cottage",
    "london",
    "road",
    "street",
    "square",
    "row",
    "place",
    "court",
    "gardens",
    "park",
    "wharf",
    "yard",
    "quarter",
}


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value.lower())).strip()


@dataclass(frozen=True)
class BuyerSearchResult:
    """Structured outcome of a deterministic buyer property search."""

    listing_category: str
    postcode: str
    has_any_at_postcode: bool
    matches: list[PropertyRecord]


class PropertyCatalog:
    """Deterministic search and lookup over properties_for_buyers.csv."""

    def __init__(self, properties: Iterable[PropertyRecord]) -> None:
        self.properties: list[PropertyRecord] = list(properties)
        self._by_id: dict[str, PropertyRecord] = {
            p.property_id.upper(): p for p in self.properties
        }

    def get_by_id(self, property_id: str) -> PropertyRecord | None:
        """Return exact property by ID (e.g. AR-0001)."""
        return self._by_id.get(property_id.strip().upper())

    def search_for_buyer(
        self,
        listing_category: str,
        postcode: str,
        max_budget_gbp: int | None = None,
        limit: int = 5,
    ) -> BuyerSearchResult:
        """Check buyer qualification & return up to `limit` available matching listings."""
        norm_pc = normalize_postcode(postcode) or postcode.strip().upper()
        cat = listing_category.strip().lower()

        at_postcode = [
            p
            for p in self.properties
            if p.listing_category == cat and p.postcode == norm_pc
        ]
        if not at_postcode:
            return BuyerSearchResult(
                listing_category=cat,
                postcode=norm_pc,
                has_any_at_postcode=False,
                matches=[],
            )

        available = [p for p in at_postcode if p.status == "available"]
        if max_budget_gbp is not None:
            available = [p for p in available if p.price_gbp <= max_budget_gbp]

        available.sort(key=lambda p: p.price_gbp)
        return BuyerSearchResult(
            listing_category=cat,
            postcode=norm_pc,
            has_any_at_postcode=True,
            matches=available[:limit],
        )

    def lookup_specific(self, query: str) -> list[PropertyRecord]:
        """Look up a specific property by ID, property_name, or address.

        Returns exact match(es) first, then substring matches, enabling disambiguation
        when a visitor names a development (e.g., 'Aldermoor Place') or street.
        """
        if not query or not query.strip():
            return []

        raw = query.strip()
        # 1. Exact ID match (e.g., AR-0008)
        id_match = re.search(r"\bAR-\d{4}\b", raw, re.IGNORECASE)
        if id_match:
            prop = self.get_by_id(id_match.group(0))
            if prop:
                return [prop]

        q_norm = _norm_text(raw)
        if not q_norm:
            return []

        # 2. Exact normalized name or address match
        exact = [
            p
            for p in self.properties
            if _norm_text(p.property_name) == q_norm or _norm_text(p.address) == q_norm
        ]
        if exact:
            return exact

        # 3. Direct phrase substring in property_name or address (require meaningful query length)
        if len(q_norm) >= 4:
            substr = [
                p
                for p in self.properties
                if q_norm in _norm_text(p.property_name)
                or q_norm in _norm_text(p.address)
            ]
            if substr:
                return substr

        # 3b. Whole-phrase match where a property name/address appears inside a longer sentence
        padded_q = f" {q_norm} "
        contained = [
            p
            for p in self.properties
            if (len(_norm_text(p.property_name)) >= 6 and f" {_norm_text(p.property_name)} " in padded_q)
            or (len(_norm_text(p.address)) >= 6 and f" {_norm_text(p.address)} " in padded_q)
        ]
        if contained:
            return contained

        # 4. Token-based fallback (all distinctive tokens appear in name + address)
        tokens = [t for t in q_norm.split() if t not in _STOPWORDS and (len(t) >= 2 or t.isdigit())]
        if not tokens:
            return []

        token_matches: list[PropertyRecord] = []
        for p in self.properties:
            haystack = f"{_norm_text(p.property_name)} {_norm_text(p.address)} {_norm_text(p.postcode)}"
            if all(re.search(rf"\b{re.escape(tok)}\b", haystack) for tok in tokens):
                token_matches.append(p)
        return token_matches

    def find_mentioned_property(self, text: str) -> str | None:
        """Return a property name/ID mentioned inside a free-form sentence, if any."""
        id_match = re.search(r"\bAR-\d{4}\b", text, re.IGNORECASE)
        if id_match:
            return id_match.group(0).upper()
        q_norm = _norm_text(text)
        if not q_norm:
            return None
        padded_q = f" {q_norm} "
        best_name: str | None = None
        best_len = 0
        for p in self.properties:
            for cand in (p.property_name, p.address):
                cand_norm = _norm_text(cand)
                if len(cand_norm) >= 6 and f" {cand_norm} " in padded_q and len(cand_norm) > best_len:
                    best_name = cand
                    best_len = len(cand_norm)
        return best_name

    @staticmethod
    def format_match_line(prop: PropertyRecord) -> str:
        """Format a single property summary line (name, bedrooms, price)."""
        bed_label = "bedroom" if prop.bedrooms == 1 else "bedrooms"
        return f"- {prop.property_name} — {prop.bedrooms} {bed_label}, {format_gbp(prop.price_gbp)}"

    @staticmethod
    def format_property_facts(prop: PropertyRecord) -> str:
        """Format full verified property details for side-question answers."""
        status_label = prop.status.replace("_", " ")
        return (
            f"{prop.property_name} ({prop.property_id}) at {prop.address}, {prop.town} {prop.postcode}: "
            f"{prop.property_type} ({prop.listing_category.replace('_', '-')}), "
            f"{prop.bedrooms} bedrooms, {prop.bathrooms} bathrooms, "
            f"priced at {format_gbp(prop.price_gbp)}, status: {status_label}, "
            f"managed by our {prop.branch} branch. {prop.description}"
        )


class SellerCoverageChecker:
    """Deterministic postcode membership check against seller_coverage.csv."""

    def __init__(self, covered_postcodes: Iterable[str]) -> None:
        self.covered_postcodes: set[str] = {
            normalize_postcode(pc) or pc.strip().upper()
            for pc in covered_postcodes
        }

    def is_covered(self, postcode: str) -> bool:
        norm = normalize_postcode(postcode)
        return bool(norm and norm in self.covered_postcodes)


class FaqRagIndex:
    """FAISS vector store retriever over faq.csv."""

    def __init__(self, faqs: list[FaqEntry], embeddings: Embeddings) -> None:
        self.faqs = faqs
        self._embeddings = embeddings
        docs = [
            Document(
                page_content=f"Category: {entry.category}\nQuestion: {entry.question}\nAnswer: {entry.answer}",
                metadata={
                    "faq_id": entry.faq_id,
                    "category": entry.category,
                    "question": entry.question,
                    "answer": entry.answer,
                },
            )
            for entry in faqs
        ]
        self._vectorstore = FAISS.from_documents(docs, embeddings)

    def retrieve(self, query: str, k: int = 3, max_l2_distance: float = 1.05) -> list[tuple[FaqEntry, float]]:
        """Retrieve top-k relevant FAQ entries filtered by L2 distance threshold."""
        if not query or not query.strip():
            return []
        results = self._vectorstore.similarity_search_with_score(query.strip(), k=k)
        matched: list[tuple[FaqEntry, float]] = []
        for doc, distance in results:
            if distance <= max_l2_distance:
                matched.append(
                    (
                        FaqEntry(
                            faq_id=doc.metadata["faq_id"],
                            category=doc.metadata["category"],
                            question=doc.metadata["question"],
                            answer=doc.metadata["answer"],
                        ),
                        float(distance),
                    )
                )
        return matched


@dataclass
class AgencyKnowledgeBase:
    """Container holding all agency data tools."""

    catalog: PropertyCatalog
    seller_coverage: SellerCoverageChecker
    faq_rag: FaqRagIndex

    @classmethod
    def from_data_dir(cls, data_dir: Path, embeddings: Embeddings) -> "AgencyKnowledgeBase":
        faqs = load_faqs(data_dir)
        properties = load_properties(data_dir)
        seller_pcs = load_seller_coverage(data_dir)
        return cls(
            catalog=PropertyCatalog(properties),
            seller_coverage=SellerCoverageChecker(seller_pcs),
            faq_rag=FaqRagIndex(faqs, embeddings),
        )
