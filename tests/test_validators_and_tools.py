"""Unit tests for deterministic parsers, validators, property lookups, seller coverage, and FAQ RAG."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_core.embeddings import Embeddings

from backend.core.knowledge_base import FaqRagIndex, PropertyCatalog, SellerCoverageChecker
from backend.core.validators import normalize_postcode, parse_budget_gbp, validate_email, validate_phone

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class DeterministicHashEmbeddings(Embeddings):
    """Offline deterministic embeddings for fast unit tests without network/model downloads."""

    dim: int = 64

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().split()
        for tok in tokens:
            digest = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = digest[0] % self.dim
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def test_parse_budget_gbp_variants() -> None:
    assert parse_budget_gbp("£1.5m") == 1_500_000
    assert parse_budget_gbp("1,200,000") == 1_200_000
    assert parse_budget_gbp("1.2 million") == 1_200_000
    assert parse_budget_gbp("950k") == 950_000
    assert parse_budget_gbp("£5k") == 5_000
    assert parse_budget_gbp("My budget is around £850,000") == 850_000
    assert parse_budget_gbp("12345 High Street, budget £1.5m") == 1_500_000
    assert parse_budget_gbp("3 bedrooms") is None
    assert parse_budget_gbp("02079460100") is None
    assert parse_budget_gbp("hello") is None


def test_normalize_postcode() -> None:
    assert normalize_postcode("gu5 7aq") == "GU5 7AQ"
    assert normalize_postcode("sw33rb") == "SW3 3RB"
    assert normalize_postcode("NW3   2LS") == "NW3 2LS"
    assert normalize_postcode("not a postcode") is None


def test_validate_phone_and_email() -> None:
    assert validate_phone("07700 900123") == "07700 900123"
    assert validate_phone("+44 20 7946 0100") == "+442079460100"
    assert validate_phone("+44 (0) 7700 900 123") == "+4407700900123"
    assert validate_phone("12345") is None

    assert validate_email("Alice.Smith@Example.COM") == "alice.smith@example.com"
    assert validate_email("invalid-email") is None


def test_property_catalog_buyer_rules() -> None:
    catalog = PropertyCatalog.from_csv(DATA_DIR)

    # SW3 3RB has 4 available new-builds (cheapest 1,230,000) and 1 under_offer (AR-0005)
    res = catalog.search_for_buyer("new_build", "sw3 3rb", max_budget_gbp=3_000_000, limit=5)
    assert res.has_any_at_postcode is True
    assert len(res.matches) == 4
    assert all(p.status == "available" for p in res.matches)
    assert "AR-0005" not in {p.property_id for p in res.matches}

    # SW3 4RR has new-builds priced 2.7m and 3.565m -> budget 1.2m yields has_any=True, matches=[]
    res_budget = catalog.search_for_buyer("new_build", "SW3 4RR", max_budget_gbp=1_200_000, limit=5)
    assert res_budget.has_any_at_postcode is True
    assert res_budget.matches == []

    # SW10 2FE has resale only -> new_build search has_any_at_postcode=False
    res_none = catalog.search_for_buyer("new_build", "SW10 2FE", max_budget_gbp=2_000_000, limit=5)
    assert res_none.has_any_at_postcode is False

    # GU1 9QZ has 5 resale listings all sold -> has_any=True, matches=[]
    res_sold = catalog.search_for_buyer("resale", "GU1 9QZ", max_budget_gbp=None, limit=5)
    assert res_sold.has_any_at_postcode is True
    assert res_sold.matches == []


def test_specific_property_lookup_and_disambiguation() -> None:
    catalog = PropertyCatalog.from_csv(DATA_DIR)

    exact = catalog.lookup_specific("Rosemount House")
    assert len(exact) == 1
    assert exact[0].property_id == "AR-0008"
    assert exact[0].status == "available"

    multi = catalog.lookup_specific("Aldermoor Place")
    assert len(multi) == 5

    missing = catalog.lookup_specific("Nonexistent Mansion 999")
    assert missing == []
    assert catalog.lookup_specific("house") == []
    assert len(catalog.lookup_specific("I'm interested in Aldermoor")) == 5


def test_local_embedding_cache_id_tracks_model_name() -> None:
    from backend.core.knowledge_base import _embedding_cache_id
    from backend.core.llm import LocalSentenceTransformerEmbeddings

    emb_a = object.__new__(LocalSentenceTransformerEmbeddings)
    emb_a.model_name = "all-MiniLM-L6-v2"
    emb_b = object.__new__(LocalSentenceTransformerEmbeddings)
    emb_b.model_name = "BAAI/bge-small-en-v1.5"
    assert _embedding_cache_id(emb_a) != _embedding_cache_id(emb_b)


def test_seller_coverage_check() -> None:
    checker = SellerCoverageChecker.from_csv(DATA_DIR)
    assert checker.is_covered("gu5 7aq") is True
    assert checker.is_covered("SW3 3RB") is True
    assert checker.is_covered("E1 6AN") is False


def test_faq_rag_retrieval() -> None:
    rag = FaqRagIndex.from_csv(DATA_DIR, DeterministicHashEmbeddings())
    hits = rag.retrieve("Chelsea branch opening hours", k=3, max_l2_distance=2.0)
    assert len(hits) >= 1
    ids = [entry.faq_id for entry, _ in hits]
    assert "FAQ-023" in ids or "FAQ-014" in ids
