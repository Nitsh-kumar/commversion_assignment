"""Knowledge Base module: CSV ingestion, property lookup, seller coverage, and FAQ RAG."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Iterator

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings

from backend.core.llm import call_with_retry
from backend.core.validators import format_gbp, normalize_postcode

_AR_ID_RE = re.compile(r"\bAR-\d{4}\b", re.IGNORECASE)

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
    "im",
    "interested",
    "looking",
    "available",
    "tell",
    "me",
    "about",
    "is",
    "what",
    "how",
    "much",
    "price",
    "status",
}


def extract_property_id(text: str) -> str | None:
    """Extract an uppercase property reference ID (e.g. 'AR-0008') if present."""
    match = _AR_ID_RE.search(text)
    return match.group(0).upper() if match else None


def _norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", value.lower())).strip()


def _resolve_csv(data_dir: Path, primary: str, fallback: str) -> Path:
    """Resolve canonical CSV name with fallback to legacy workspace filename."""
    p = data_dir / primary
    if p.exists():
        return p
    alt = data_dir.parent / fallback
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Could not find {primary} in {data_dir} or {fallback} in {data_dir.parent}")


def _read_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    """Read CSV rows as stripped string dicts using stdlib csv."""
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield {k: (v or "").strip() for k, v in row.items() if k is not None}


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

    @classmethod
    def from_csv(cls, data_dir: Path) -> "PropertyCatalog":
        """Load and normalize property listings directly from data_dir."""
        path = _resolve_csv(data_dir, "properties_for_buyers.csv", "properties_for_buyers 1.csv")
        records: list[PropertyRecord] = []
        for row in _read_csv_rows(path):
            norm_pc = normalize_postcode(row["postcode"]) or row["postcode"].upper()
            records.append(
                PropertyRecord(
                    property_id=row["property_id"],
                    property_name=row["property_name"],
                    address=row["address"],
                    town=row["town"],
                    postcode=norm_pc,
                    listing_category=row["listing_category"].lower(),
                    property_type=row["property_type"],
                    bedrooms=int(row["bedrooms"]),
                    bathrooms=int(row["bathrooms"]),
                    price_gbp=int(row["price_gbp"]),
                    status=row["status"].lower(),
                    description=row["description"],
                    branch=row["branch"],
                    listed_date=row["listed_date"],
                )
            )
        return cls(records)

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
        """Check buyer qualification and return up to `limit` available matching listings."""
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

    def _iter_contained_phrase_matches(self, q_norm: str) -> Iterator[tuple[PropertyRecord, str, int]]:
        """Yield (record, matched_candidate_text, normalized_length) when a property appears in a sentence."""
        padded_q = f" {q_norm} "
        for prop in self.properties:
            for candidate in (prop.property_name, prop.address):
                cand_norm = _norm_text(candidate)
                if len(cand_norm) >= 6 and f" {cand_norm} " in padded_q:
                    yield prop, candidate, len(cand_norm)

    def lookup_specific(self, query: str) -> list[PropertyRecord]:
        """Look up a specific property by ID, property_name, or address."""
        if not query or not query.strip():
            return []

        raw = query.strip()
        prop_id = extract_property_id(raw)
        if prop_id:
            prop = self.get_by_id(prop_id)
            if prop:
                return [prop]

        q_norm = _norm_text(raw)
        if not q_norm:
            return []

        exact = [
            p
            for p in self.properties
            if _norm_text(p.property_name) == q_norm or _norm_text(p.address) == q_norm
        ]
        if exact:
            return exact

        if len(q_norm) >= 4 and q_norm not in _STOPWORDS:
            substr = [
                p
                for p in self.properties
                if q_norm in _norm_text(p.property_name)
                or q_norm in _norm_text(p.address)
            ]
            if substr:
                return substr

        contained = list(dict.fromkeys(prop for prop, _, _ in self._iter_contained_phrase_matches(q_norm)))
        if contained:
            return contained

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
        prop_id = extract_property_id(text)
        if prop_id:
            return prop_id
        q_norm = _norm_text(text)
        if not q_norm:
            return None
        best_name: str | None = None
        best_len = 0
        for _, candidate, cand_len in self._iter_contained_phrase_matches(q_norm):
            if cand_len > best_len:
                best_name = candidate
                best_len = cand_len
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
            norm for pc in covered_postcodes if (norm := normalize_postcode(pc))
        }

    @classmethod
    def from_csv(cls, data_dir: Path) -> "SellerCoverageChecker":
        """Load covered seller postcodes directly from data_dir."""
        path = _resolve_csv(data_dir, "seller_coverage.csv", "seller_coverage 1.csv")
        return cls(row["postcode"] for row in _read_csv_rows(path))

    def is_covered(self, postcode: str) -> bool:
        norm = normalize_postcode(postcode)
        return bool(norm and norm in self.covered_postcodes)


def _default_max_l2_distance(embeddings: Embeddings) -> float:
    """Select a calibrated L2 distance threshold for the active embedding model."""
    cls_name = type(embeddings).__name__.lower()
    model_name = str(getattr(embeddings, "model", getattr(embeddings, "model_name", ""))).lower()
    if "google" in cls_name or "gemini" in model_name:
        return 0.65
    return 1.05


def _embedding_cache_id(embeddings: Embeddings) -> str | None:
    """Return a stable cache identifier for real embedding backends (None for test doubles)."""
    model_name = getattr(embeddings, "model", None) or getattr(embeddings, "model_name", None)
    if not model_name and "sentence" in type(embeddings).__name__.lower():
        model_name = "sentence-transformers"
    if not model_name:
        return None
    return f"{type(embeddings).__qualname__}:{model_name}"


def _build_cached_faq_vectorstore(
    texts: list[str],
    metadatas: list[dict[str, str]],
    embeddings: Embeddings,
    cache_dir: Path | None,
) -> FAISS:
    """Embed FAQ texts once per (model, content) and reuse vectors from JSON cache."""
    cache_id = _embedding_cache_id(embeddings)
    cache_file: Path | None = None
    if cache_dir is not None and cache_id is not None:
        digest = hashlib.sha256(
            (cache_id + "\n" + "\n---\n".join(texts)).encode("utf-8")
        ).hexdigest()[:16]
        cache_file = cache_dir / f"faq_{digest}.json"
        if cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                vectors = payload.get("vectors")
                if isinstance(vectors, list) and len(vectors) == len(texts):
                    return FAISS.from_embeddings(
                        text_embeddings=list(zip(texts, vectors)),
                        embedding=embeddings,
                        metadatas=metadatas,
                    )
            except (OSError, ValueError, TypeError):
                pass

    vectors = call_with_retry(lambda: embeddings.embed_documents(texts))
    if cache_file is not None:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({"model": cache_id, "vectors": vectors}),
                encoding="utf-8",
            )
        except OSError:
            pass

    return FAISS.from_embeddings(
        text_embeddings=list(zip(texts, vectors)),
        embedding=embeddings,
        metadatas=metadatas,
    )


class FaqRagIndex:
    """FAISS vector store retriever over faq.csv."""

    def __init__(
        self,
        faqs: list[FaqEntry],
        embeddings: Embeddings,
        default_max_l2_distance: float | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.faqs = faqs
        self._embeddings = embeddings
        self.default_max_l2_distance = (
            default_max_l2_distance
            if default_max_l2_distance is not None
            else _default_max_l2_distance(embeddings)
        )
        texts = [
            f"Category: {entry.category}\nQuestion: {entry.question}\nAnswer: {entry.answer}"
            for entry in faqs
        ]
        metadatas = [
            {
                "faq_id": entry.faq_id,
                "category": entry.category,
                "question": entry.question,
                "answer": entry.answer,
            }
            for entry in faqs
        ]
        self._vectorstore = _build_cached_faq_vectorstore(texts, metadatas, embeddings, cache_dir)

    @classmethod
    def from_csv(
        cls,
        data_dir: Path,
        embeddings: Embeddings,
        default_max_l2_distance: float | None = None,
    ) -> "FaqRagIndex":
        """Load FAQ rows from data_dir and build (or reuse cached) FAISS index."""
        path = _resolve_csv(data_dir, "faq.csv", "faq 1.csv")
        faqs = [
            FaqEntry(
                faq_id=row["faq_id"],
                category=row["category"],
                question=row["question"],
                answer=row["answer"],
            )
            for row in _read_csv_rows(path)
        ]
        cache_dir = data_dir.parent / ".cache" / "faq_embeddings"
        return cls(
            faqs,
            embeddings,
            default_max_l2_distance=default_max_l2_distance,
            cache_dir=cache_dir,
        )

    def retrieve(
        self,
        query: str,
        k: int = 3,
        max_l2_distance: float | None = None,
    ) -> list[tuple[FaqEntry, float]]:
        """Retrieve top-k relevant FAQ entries filtered by L2 distance threshold."""
        if not query or not query.strip():
            return []
        threshold = self.default_max_l2_distance if max_l2_distance is None else max_l2_distance
        stripped = query.strip()
        results = call_with_retry(
            lambda: self._vectorstore.similarity_search_with_score(stripped, k=k)
        )
        matched: list[tuple[FaqEntry, float]] = []
        for doc, distance in results:
            if distance <= threshold:
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
    """Deep Knowledge Base module owning CSV ingestion, catalog, seller coverage, and FAQ RAG."""

    catalog: PropertyCatalog
    seller_coverage: SellerCoverageChecker
    faq_rag: FaqRagIndex

    @classmethod
    def from_data_dir(
        cls,
        data_dir: Path,
        embeddings: Embeddings,
        faq_max_l2_distance: float | None = None,
    ) -> "AgencyKnowledgeBase":
        return cls(
            catalog=PropertyCatalog.from_csv(data_dir),
            seller_coverage=SellerCoverageChecker.from_csv(data_dir),
            faq_rag=FaqRagIndex.from_csv(
                data_dir,
                embeddings,
                default_max_l2_distance=faq_max_l2_distance,
            ),
        )
