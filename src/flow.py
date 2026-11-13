"""Deterministic conversation state machine reproducing flowchart.png."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.config import Settings
from src.data_loader import PropertyRecord
from src.tools import PropertyCatalog, SellerCoverageChecker
from src.validators import (
    format_gbp,
    normalize_postcode,
    parse_budget_gbp,
    validate_email,
    validate_phone,
)


class Step(str, Enum):
    """Exact flowchart step identifiers across Start and Lanes A-E."""

    ASK_NAME = "ASK_NAME"
    BUY_OR_SELL = "BUY_OR_SELL"
    SPECIFIC_PROPERTY = "SPECIFIC_PROPERTY"
    NEW_OR_RESALE = "NEW_OR_RESALE"

    # Lane A: New-build buyer
    A1_ASK_BUDGET = "A1_ASK_BUDGET"
    A3_ASK_POSTCODE = "A3_ASK_POSTCODE"

    # Lane B: Resale buyer
    B1_ASK_BUDGET = "B1_ASK_BUDGET"
    B2_ASK_POSTCODE = "B2_ASK_POSTCODE"

    # Lane C: Specific property buyer
    C1_ASK_PROPERTY = "C1_ASK_PROPERTY"
    C1_RETRY_PROPERTY = "C1_RETRY_PROPERTY"
    C1_DISAMBIGUATE = "C1_DISAMBIGUATE"
    C5_ASK_BUDGET = "C5_ASK_BUDGET"

    # Lane D: Seller
    D1_ASK_POSTCODE = "D1_ASK_POSTCODE"

    # Lane E: Contact details & close
    E1_ASK_PHONE = "E1_ASK_PHONE"
    E2_ASK_EMAIL = "E2_ASK_EMAIL"
    E3_REASSIST = "E3_REASSIST"

    ENDED = "ENDED"


@dataclass
class ExtractedTurn:
    """Normalized signals extracted from a single user message."""

    visitor_name: str | None = None
    intent: str | None = None  # "buy" | "sell"
    has_specific_property: bool | None = None
    specific_property_query: str | None = None
    listing_category: str | None = None  # "new_build" | "resale"
    budget_text: str | None = None
    postcode_text: str | None = None
    phone_text: str | None = None
    email_text: str | None = None
    yes_no: str | None = None  # "yes" | "no"
    mind_change_target: str | None = None  # "buy" | "sell" | "new_build" | "resale" | "specific_property"
    selection_index: int | None = None  # 1-based choice during C1_DISAMBIGUATE


@dataclass
class ConversationData:
    """Mutable structured state held across the conversation."""

    current_step: Step = Step.ASK_NAME
    visitor_name: str | None = None
    intent: str | None = None
    listing_category: str | None = None
    budget_gbp: int | None = None
    postcode: str | None = None
    selected_property_id: str | None = None
    specific_property_category: str | None = None
    candidate_property_ids: list[str] = field(default_factory=list)
    matched_property_ids: list[str] = field(default_factory=list)
    qualified: bool = False
    phone: str | None = None
    email: str | None = None
    closing_code: str | None = None  # "M1" | "M2" | "M3" | "GOODBYE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_step": self.current_step.value,
            "visitor_name": self.visitor_name,
            "intent": self.intent,
            "listing_category": self.listing_category,
            "budget_gbp": self.budget_gbp,
            "postcode": self.postcode,
            "selected_property_id": self.selected_property_id,
            "specific_property_category": self.specific_property_category,
            "candidate_property_ids": list(self.candidate_property_ids),
            "matched_property_ids": list(self.matched_property_ids),
            "qualified": self.qualified,
            "phone": self.phone,
            "email": self.email,
            "closing_code": self.closing_code,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ConversationData":
        if not raw:
            return cls()
        return cls(
            current_step=Step(raw.get("current_step", Step.ASK_NAME.value)),
            visitor_name=raw.get("visitor_name"),
            intent=raw.get("intent"),
            listing_category=raw.get("listing_category"),
            budget_gbp=raw.get("budget_gbp"),
            postcode=raw.get("postcode"),
            selected_property_id=raw.get("selected_property_id"),
            specific_property_category=raw.get("specific_property_category"),
            candidate_property_ids=list(raw.get("candidate_property_ids") or []),
            matched_property_ids=list(raw.get("matched_property_ids") or []),
            qualified=bool(raw.get("qualified", False)),
            phone=raw.get("phone"),
            email=raw.get("email"),
            closing_code=raw.get("closing_code"),
        )


def _closing_message(code: str, office_phone: str) -> str:
    reasons = {
        "M1": "Sorry, we don't cater to new-build properties under £1 million.",
        "M2": "Sorry, we don't cater to the postcode you provided.",
        "M3": "Sorry, we have no available homes matching your requirements right now.",
    }
    reason = reasons[code]
    return f"{reason} Please call our head office on {office_phone}. Goodbye."


def prompt_for_step(step: Step, data: ConversationData) -> str:
    """Return the canonical question for a given flowchart step."""
    name_prefix = f"{data.visitor_name}, " if data.visitor_name else ""
    prompts = {
        Step.ASK_NAME: "How may I help you? Can I get your name?",
        Step.BUY_OR_SELL: f"Thanks, {data.visitor_name}. Are you looking to buy or sell a property?"
        if data.visitor_name
        else "How may I help you? Are you looking to buy or sell a property?",
        Step.SPECIFIC_PROPERTY: f"{name_prefix}do you have a specific property in mind?",
        Step.NEW_OR_RESALE: f"{name_prefix}are you looking for a new-build or a resale home?",
        Step.A1_ASK_BUDGET: f"{name_prefix}what is your budget?",
        Step.A3_ASK_POSTCODE: f"{name_prefix}what is the postcode of your location of interest?",
        Step.B1_ASK_BUDGET: f"{name_prefix}what is your budget?",
        Step.B2_ASK_POSTCODE: f"{name_prefix}what is the postcode of your location of interest?",
        Step.C1_ASK_PROPERTY: f"{name_prefix}which property is it?",
        Step.C1_RETRY_PROPERTY: (
            f"I couldn't find a matching property, {data.visitor_name}. Could you share the property name or address once more?"
            if data.visitor_name
            else "I couldn't find a matching property. Could you share the property name or address once more?"
        ),
        Step.C1_DISAMBIGUATE: "Which of those properties did you mean?",
        Step.C5_ASK_BUDGET: "Sorry, that one's gone. I can help you find a similar home. What's your budget?",
        Step.D1_ASK_POSTCODE: f"{name_prefix}what is your postcode?",
        Step.E1_ASK_PHONE: f"{name_prefix}can I get your phone number?",
        Step.E2_ASK_EMAIL: f"{name_prefix}can I get your email address?",
        Step.E3_REASSIST: "Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?",
        Step.ENDED: "Thank you for chatting with us. Goodbye.",
    }
    return prompts[step]


def _reset_journey_slots(data: ConversationData) -> None:
    """Reset journey-specific fields while preserving visitor_name, phone, and email."""
    data.intent = None
    data.listing_category = None
    data.budget_gbp = None
    data.postcode = None
    data.selected_property_id = None
    data.specific_property_category = None
    data.candidate_property_ids = []
    data.matched_property_ids = []
    data.qualified = False
    data.closing_code = None


def _enter_lane_e(data: ConversationData, prefix_parts: list[str]) -> str:
    """Transition into Lane E, preserving previously collected contact details if any."""
    data.qualified = True
    if data.phone and data.email:
        data.current_step = Step.E3_REASSIST
        prefix_parts.append(
            f"I already have your contact details on file ({data.phone} / {data.email}). "
            "Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?"
        )
        return "\n\n".join(prefix_parts)
    if data.phone and not data.email:
        data.current_step = Step.E2_ASK_EMAIL
        prefix_parts.append(prompt_for_step(Step.E2_ASK_EMAIL, data))
        return "\n\n".join(prefix_parts)
    data.current_step = Step.E1_ASK_PHONE
    prefix_parts.append(prompt_for_step(Step.E1_ASK_PHONE, data))
    return "\n\n".join(prefix_parts)


def _evaluate_buyer_search(
    data: ConversationData,
    catalog: PropertyCatalog,
    settings: Settings,
    prefix_parts: list[str],
    listing_category: str,
    max_budget_gbp: int | None,
) -> str:
    """Run deterministic buyer property search (A4-A6 or B3-B5) and update state."""
    assert data.postcode is not None
    result = catalog.search_for_buyer(
        listing_category=listing_category,
        postcode=data.postcode,
        max_budget_gbp=max_budget_gbp,
        limit=settings.max_property_matches,
    )
    if not result.has_any_at_postcode:
        data.current_step = Step.ENDED
        data.qualified = False
        data.closing_code = "M2"
        prefix_parts.append(_closing_message("M2", settings.office_phone))
        return "\n\n".join(prefix_parts)

    if not result.matches:
        data.current_step = Step.ENDED
        data.qualified = False
        data.closing_code = "M3"
        prefix_parts.append(_closing_message("M3", settings.office_phone))
        return "\n\n".join(prefix_parts)

    data.matched_property_ids = [p.property_id for p in result.matches]
    lines = [catalog.format_match_line(p) for p in result.matches]
    heading = (
        f"Great news{f', {data.visitor_name}' if data.visitor_name else ''}! "
        f"Here {'is a matching available property' if len(lines) == 1 else f'are {len(lines)} matching available properties'} in {data.postcode}:"
    )
    prefix_parts.append(heading + "\n" + "\n".join(lines))
    return _enter_lane_e(data, prefix_parts)


def _handle_specific_property_lookup(
    data: ConversationData,
    query: str,
    catalog: PropertyCatalog,
    prefix_parts: list[str],
    is_retry: bool,
) -> tuple[str | None, bool]:
    """Resolve Lane C lookup (C2-C5). Returns (response_if_terminal_for_turn, advanced)."""
    matches = catalog.lookup_specific(query)

    if not matches:
        if is_retry:
            data.current_step = Step.NEW_OR_RESALE
            prefix_parts.append(
                "I still couldn't find that property in our listings. "
                "Let's check what else we have—are you looking for a new-build or a resale home?"
            )
            return "\n\n".join(prefix_parts), False
        data.current_step = Step.C1_RETRY_PROPERTY
        prefix_parts.append(prompt_for_step(Step.C1_RETRY_PROPERTY, data))
        return "\n\n".join(prefix_parts), False

    if len(matches) > 1:
        shown = matches[:8]
        data.candidate_property_ids = [p.property_id for p in shown]
        data.current_step = Step.C1_DISAMBIGUATE
        options = "\n".join(
            f"{idx}. {p.property_name} ({p.address}, {p.postcode})"
            for idx, p in enumerate(shown, start=1)
        )
        prefix_parts.append(
            f"I found {len(matches)} matching properties. Which one did you mean?\n{options}"
        )
        return "\n\n".join(prefix_parts), False

    prop = matches[0]
    return _apply_selected_property(data, prop, prefix_parts), True


def _apply_selected_property(
    data: ConversationData,
    prop: PropertyRecord,
    prefix_parts: list[str],
) -> str | None:
    """Process a uniquely resolved Lane C property (C3-C5)."""
    data.selected_property_id = prop.property_id
    data.candidate_property_ids = []
    data.specific_property_category = prop.listing_category

    if prop.status == "available":
        bed_label = "bedroom" if prop.bedrooms == 1 else "bedrooms"
        prefix_parts.append(
            f"{prop.property_name} ({prop.postcode}) is available! "
            f"It is a {prop.property_type.lower()} with {prop.bedrooms} {bed_label}, "
            f"priced at {format_gbp(prop.price_gbp)}."
        )
        return _enter_lane_e(data, prefix_parts)

    # C5: Sold or under offer
    status_text = "under offer" if prop.status == "under_offer" else "sold"
    data.current_step = Step.C5_ASK_BUDGET
    prefix_parts.append(
        f"Sorry, {prop.property_name} is currently {status_text}. "
        "I can help you find a similar home. What's your budget?"
    )
    return None  # Caller may advance immediately if user also gave a budget


def apply_mind_change(data: ConversationData, target: str) -> None:
    """Handle mid-conversation intent or category switch while keeping visitor details."""
    preserved_name = data.visitor_name
    preserved_phone = data.phone
    preserved_email = data.email

    if target == "sell":
        _reset_journey_slots(data)
        data.intent = "sell"
        data.current_step = Step.D1_ASK_POSTCODE
    elif target == "buy":
        _reset_journey_slots(data)
        data.intent = "buy"
        data.current_step = Step.SPECIFIC_PROPERTY
    elif target == "new_build":
        _reset_journey_slots(data)
        data.intent = "buy"
        data.listing_category = "new_build"
        data.current_step = Step.A1_ASK_BUDGET
    elif target == "resale":
        _reset_journey_slots(data)
        data.intent = "buy"
        data.listing_category = "resale"
        data.current_step = Step.B1_ASK_BUDGET
    elif target == "specific_property":
        _reset_journey_slots(data)
        data.intent = "buy"
        data.current_step = Step.C1_ASK_PROPERTY

    data.visitor_name = preserved_name
    data.phone = preserved_phone
    data.email = preserved_email


def advance_conversation(
    data: ConversationData,
    extracted: ExtractedTurn,
    raw_user_message: str,
    catalog: PropertyCatalog,
    seller_coverage: SellerCoverageChecker,
    settings: Settings,
) -> str:
    """Run deterministic flowchart transitions for one user turn and return flow response."""
    if data.current_step == Step.ENDED:
        return "This conversation has ended. Please start a new chat if you need further assistance."

    prefix_parts: list[str] = []

    # Opportunistically store visitor name if provided at any early step
    if not data.visitor_name and extracted.visitor_name:
        data.visitor_name = extracted.visitor_name.strip()

    # Handle mid-conversation mind changes (not applicable during E3_REASSIST where yes/no rules apply)
    if extracted.mind_change_target and data.current_step != Step.E3_REASSIST:
        apply_mind_change(data, extracted.mind_change_target)
        # Clear extracted.intent if it just echoed the switch so we don't loop weirdly
        if extracted.mind_change_target in {"buy", "sell"}:
            extracted.intent = None
        if extracted.mind_change_target in {"new_build", "resale"}:
            extracted.listing_category = None

    # Allow multi-step progression within a single turn when user supplies multiple valid slots
    for _ in range(8):
        step = data.current_step

        if step == Step.ASK_NAME:
            name = data.visitor_name or extracted.visitor_name
            if not name:
                return prompt_for_step(Step.ASK_NAME, data)
            data.visitor_name = name.strip()
            data.current_step = Step.BUY_OR_SELL
            # Continue loop in case user also gave buy/sell/property in their first message
            if extracted.intent or extracted.specific_property_query:
                continue
            return prompt_for_step(Step.BUY_OR_SELL, data)

        if step == Step.BUY_OR_SELL:
            if extracted.specific_property_query and extracted.intent != "sell":
                extracted.intent = "buy"

            if extracted.intent == "sell":
                data.intent = "sell"
                data.current_step = Step.D1_ASK_POSTCODE
                if extracted.postcode_text:
                    continue
                return prompt_for_step(Step.D1_ASK_POSTCODE, data)

            if extracted.intent == "buy":
                data.intent = "buy"
                # PDF rule: If the visitor names a property in their first message, go straight to lane C
                if extracted.specific_property_query:
                    data.current_step = Step.C1_ASK_PROPERTY
                    continue
                if extracted.listing_category in {"new_build", "resale"}:
                    data.listing_category = extracted.listing_category
                    data.current_step = (
                        Step.A1_ASK_BUDGET if data.listing_category == "new_build" else Step.B1_ASK_BUDGET
                    )
                    if extracted.budget_text:
                        continue
                    return prompt_for_step(data.current_step, data)
                data.current_step = Step.SPECIFIC_PROPERTY
                if extracted.has_specific_property is not None:
                    continue
                return prompt_for_step(Step.SPECIFIC_PROPERTY, data)

            return prompt_for_step(Step.BUY_OR_SELL, data)

        if step == Step.SPECIFIC_PROPERTY:
            if extracted.specific_property_query or extracted.has_specific_property is True:
                data.current_step = Step.C1_ASK_PROPERTY
                if extracted.specific_property_query:
                    continue
                return prompt_for_step(Step.C1_ASK_PROPERTY, data)

            if extracted.has_specific_property is False or extracted.listing_category in {"new_build", "resale"}:
                data.current_step = Step.NEW_OR_RESALE
                if extracted.listing_category in {"new_build", "resale"}:
                    continue
                return prompt_for_step(Step.NEW_OR_RESALE, data)

            return prompt_for_step(Step.SPECIFIC_PROPERTY, data)

        if step == Step.NEW_OR_RESALE:
            if extracted.listing_category == "new_build":
                data.listing_category = "new_build"
                data.current_step = Step.A1_ASK_BUDGET
                if extracted.budget_text:
                    continue
                return prompt_for_step(Step.A1_ASK_BUDGET, data)
            if extracted.listing_category == "resale":
                data.listing_category = "resale"
                data.current_step = Step.B1_ASK_BUDGET
                if extracted.budget_text:
                    continue
                return prompt_for_step(Step.B1_ASK_BUDGET, data)
            return prompt_for_step(Step.NEW_OR_RESALE, data)

        # ---------------- LANE A: NEW-BUILD BUYER ----------------
        if step == Step.A1_ASK_BUDGET:
            budget = parse_budget_gbp(extracted.budget_text or raw_user_message)
            if budget is None:
                return (
                    "Sorry, I didn't catch a valid budget amount (for example, £1.2m or £950,000). "
                    "What is your budget?"
                )
            data.budget_gbp = budget
            extracted.budget_text = None  # consumed
            if budget < settings.new_build_min_budget_gbp:
                data.current_step = Step.ENDED
                data.qualified = False
                data.closing_code = "M1"
                return _closing_message("M1", settings.office_phone)

            data.current_step = Step.A3_ASK_POSTCODE
            if extracted.postcode_text:
                continue
            return prompt_for_step(Step.A3_ASK_POSTCODE, data)

        if step == Step.A3_ASK_POSTCODE:
            postcode = normalize_postcode(extracted.postcode_text or raw_user_message)
            if not postcode:
                return (
                    "Please enter a valid full UK postcode (for example, SW3 3RB or NW3 2LS). "
                    "Postcode of your location of interest?"
                )
            data.postcode = postcode
            return _evaluate_buyer_search(
                data=data,
                catalog=catalog,
                settings=settings,
                prefix_parts=prefix_parts,
                listing_category="new_build",
                max_budget_gbp=data.budget_gbp,
            )

        # ---------------- LANE B: RESALE BUYER ----------------
        if step == Step.B1_ASK_BUDGET:
            budget = parse_budget_gbp(extracted.budget_text or raw_user_message)
            if budget is None:
                return (
                    "Sorry, I didn't catch a valid budget amount (for example, £850,000 or £1.5m). "
                    "What is your budget?"
                )
            data.budget_gbp = budget
            extracted.budget_text = None  # consumed
            data.current_step = Step.B2_ASK_POSTCODE
            if extracted.postcode_text:
                continue
            return prompt_for_step(Step.B2_ASK_POSTCODE, data)

        if step == Step.B2_ASK_POSTCODE:
            postcode = normalize_postcode(extracted.postcode_text or raw_user_message)
            if not postcode:
                return (
                    "Please enter a valid full UK postcode (for example, SW10 9RR or GU5 7AQ). "
                    "Postcode of your location of interest?"
                )
            data.postcode = postcode
            # Resale budget is recorded only and NOT used as a filter
            return _evaluate_buyer_search(
                data=data,
                catalog=catalog,
                settings=settings,
                prefix_parts=prefix_parts,
                listing_category="resale",
                max_budget_gbp=None,
            )

        # ---------------- LANE C: SPECIFIC PROPERTY ----------------
        if step in {Step.C1_ASK_PROPERTY, Step.C1_RETRY_PROPERTY}:
            query = (extracted.specific_property_query or raw_user_message).strip()
            extracted.specific_property_query = None
            resp, _ = _handle_specific_property_lookup(
                data=data,
                query=query,
                catalog=catalog,
                prefix_parts=prefix_parts,
                is_retry=(step == Step.C1_RETRY_PROPERTY),
            )
            if resp is not None:
                return resp
            # C5 was entered (sold/under_offer); if budget was also provided, consume it
            if extracted.budget_text:
                continue
            return "\n\n".join(prefix_parts)

        if step == Step.C1_DISAMBIGUATE:
            candidates = [
                p for pid in data.candidate_property_ids if (p := catalog.get_by_id(pid)) is not None
            ]
            chosen: PropertyRecord | None = None
            if extracted.selection_index is not None and 1 <= extracted.selection_index <= len(candidates):
                chosen = candidates[extracted.selection_index - 1]
            else:
                q = (extracted.specific_property_query or raw_user_message).strip()
                if q.isdigit() and 1 <= int(q) <= len(candidates):
                    chosen = candidates[int(q) - 1]
                else:
                    sub_matches = [
                        p
                        for p in catalog.lookup_specific(q)
                        if p.property_id in data.candidate_property_ids
                    ]
                    if len(sub_matches) == 1:
                        chosen = sub_matches[0]
                    elif len(sub_matches) > 1:
                        data.candidate_property_ids = [p.property_id for p in sub_matches[:8]]
                        options = "\n".join(
                            f"{idx}. {p.property_name} ({p.address}, {p.postcode})"
                            for idx, p in enumerate(sub_matches[:8], start=1)
                        )
                        return f"Several options still match—could you pick one by number or full name?\n{options}"

            if chosen is None:
                options = "\n".join(
                    f"{idx}. {p.property_name} ({p.address}, {p.postcode})"
                    for idx, p in enumerate(candidates, start=1)
                )
                return f"I didn't catch which property you meant. Please reply with the number or full property name:\n{options}"

            resp = _apply_selected_property(data, chosen, prefix_parts)
            if resp is not None:
                return resp
            return "\n\n".join(prefix_parts)

        if step == Step.C5_ASK_BUDGET:
            budget = parse_budget_gbp(extracted.budget_text or raw_user_message)
            if budget is None:
                return (
                    "Sorry, I didn't catch a valid budget amount (for example, £1.2m or £900k). "
                    "What's your budget?"
                )
            data.budget_gbp = budget
            extracted.budget_text = None
            cat = data.specific_property_category or "resale"
            data.listing_category = cat

            if cat == "new_build":
                if budget < settings.new_build_min_budget_gbp:
                    data.current_step = Step.ENDED
                    data.qualified = False
                    data.closing_code = "M1"
                    return _closing_message("M1", settings.office_phone)
                data.current_step = Step.A3_ASK_POSTCODE
                if extracted.postcode_text:
                    continue
                return prompt_for_step(Step.A3_ASK_POSTCODE, data)

            data.current_step = Step.B2_ASK_POSTCODE
            if extracted.postcode_text:
                continue
            return prompt_for_step(Step.B2_ASK_POSTCODE, data)

        # ---------------- LANE D: SELLER ----------------
        if step == Step.D1_ASK_POSTCODE:
            postcode = normalize_postcode(extracted.postcode_text or raw_user_message)
            if not postcode:
                return (
                    "Please provide a valid full UK postcode (for example, GU5 7AQ or SW3 3RB). "
                    "What is your postcode?"
                )
            data.postcode = postcode
            if not seller_coverage.is_covered(postcode):
                data.current_step = Step.ENDED
                data.qualified = False
                data.closing_code = "M2"
                return _closing_message("M2", settings.office_phone)

            prefix_parts.append(
                f"Good news{f', {data.visitor_name}' if data.visitor_name else ''}—we cover {postcode} for property sales!"
            )
            return _enter_lane_e(data, prefix_parts)

        # ---------------- LANE E: CONTACT DETAILS & CLOSE ----------------
        if step == Step.E1_ASK_PHONE:
            phone = validate_phone(extracted.phone_text or raw_user_message)
            if not phone:
                return (
                    "That doesn't look like a valid phone number. "
                    "Could you share a UK or international contact number (at least 10 digits)?"
                )
            data.phone = phone
            extracted.phone_text = None
            if data.email:
                data.current_step = Step.E3_REASSIST
                return prompt_for_step(Step.E3_REASSIST, data)
            data.current_step = Step.E2_ASK_EMAIL
            if extracted.email_text:
                continue
            return prompt_for_step(Step.E2_ASK_EMAIL, data)

        if step == Step.E2_ASK_EMAIL:
            email = validate_email(extracted.email_text or raw_user_message)
            if not email:
                return (
                    "That doesn't look like a valid email address. "
                    "Could you please provide your email address (e.g. name@example.com)?"
                )
            data.email = email
            extracted.email_text = None
            data.current_step = Step.E3_REASSIST
            return prompt_for_step(Step.E3_REASSIST, data)

        if step == Step.E3_REASSIST:
            # Reassist if yes or visitor directly states a new buy/sell/property request
            if extracted.yes_no == "yes" or extracted.intent or extracted.specific_property_query:
                preserved_name = data.visitor_name
                preserved_phone = data.phone
                preserved_email = data.email
                _reset_journey_slots(data)
                data.visitor_name = preserved_name
                data.phone = preserved_phone
                data.email = preserved_email
                data.current_step = Step.BUY_OR_SELL
                if extracted.intent or extracted.specific_property_query:
                    continue
                return "How can I help you? Are you looking to buy or sell a property?"

            if extracted.yes_no == "no":
                data.current_step = Step.ENDED
                data.closing_code = "GOODBYE"
                return prompt_for_step(Step.ENDED, data)

            return "Is there anything else I can help you with?"

    return prompt_for_step(data.current_step, data)
