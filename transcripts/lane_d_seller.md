# Lane D — Seller (Covered vs Reassist) & Uncovered (M2)

**Assistant:** Hello! Welcome to Ashcombe & Rowe Estates. How may I help you? Can I get your name?

**Visitor:** Ivy

**Assistant:** Thanks, Ivy. Are you looking to buy or sell a property?

> _Step: `ASK_NAME` -> `BUY_OR_SELL` | Tools: `[]` | Latency: `2.99 ms`_

**Visitor:** I want to sell my house

**Assistant:** Ivy, what is your postcode?

> _Step: `BUY_OR_SELL` -> `D1_ASK_POSTCODE` | Tools: `[]` | Latency: `3.03 ms`_

**Visitor:** gu5 7aq

**Assistant:** Good news, Ivy—we cover GU5 7AQ for property sales!

Ivy, can I get your phone number?

> _Step: `D1_ASK_POSTCODE` -> `E1_ASK_PHONE` | Tools: `['seller_coverage_check']` | Latency: `3.38 ms`_

**Visitor:** 07700 900222

**Assistant:** Ivy, can I get your email address?

> _Step: `E1_ASK_PHONE` -> `E2_ASK_EMAIL` | Tools: `[]` | Latency: `3.82 ms`_

**Visitor:** ivy@example.com

**Assistant:** Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?

> _Step: `E2_ASK_EMAIL` -> `E3_REASSIST` | Tools: `[]` | Latency: `5.92 ms`_

**Visitor:** No

**Assistant:** Thank you for chatting with us. Goodbye.

> _Step: `E3_REASSIST` -> `ENDED` | Tools: `[]` | Latency: `3.17 ms`_

---
## Final Qualification State
- `qualified`: `True`
- `closing_code`: `GOODBYE`
- `visitor_name`: `Ivy`
- `intent`: `sell`
- `listing_category`: `None`
- `budget_gbp`: `None`
- `postcode`: `GU5 7AQ`
- `phone`: `07700 900222`
- `email`: `ivy@example.com`
