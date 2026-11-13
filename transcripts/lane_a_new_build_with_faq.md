# Lane A — New-Build Buyer (Qualified) + Mid-Flow FAQ

**Assistant:** Hello! Welcome to Ashcombe & Rowe Estates. How may I help you? Can I get your name?

**Visitor:** Alice

**Assistant:** Thanks, Alice. Are you looking to buy or sell a property?

> _Step: `ASK_NAME` -> `BUY_OR_SELL` | Tools: `[]` | Latency: `5.73 ms`_

**Visitor:** I am looking to buy a home

**Assistant:** Alice, do you have a specific property in mind?

> _Step: `BUY_OR_SELL` -> `SPECIFIC_PROPERTY` | Tools: `[]` | Latency: `2.71 ms`_

**Visitor:** No

**Assistant:** Alice, are you looking for a new-build or a resale home?

> _Step: `SPECIFIC_PROPERTY` -> `NEW_OR_RESALE` | Tools: `[]` | Latency: `3.91 ms`_

**Visitor:** New-build

**Assistant:** Alice, what is your budget?

> _Step: `NEW_OR_RESALE` -> `A1_ASK_BUDGET` | Tools: `[]` | Latency: `15.1 ms`_

**Visitor:** What are your Chelsea branch opening hours?

**Assistant:** Chelsea branch: Monday to Friday 9:00 am - 7:00 pm, Saturday 10:00 am - 5:00 pm, Sunday 11:00 am - 3:00 pm.

To continue where we left off — Alice, what is your budget?

> _Step: `A1_ASK_BUDGET` -> `A1_ASK_BUDGET` | Tools: `['faq_rag']` | Latency: `38.23 ms`_

**Visitor:** £1.5m

**Assistant:** Alice, what is the postcode of your location of interest?

> _Step: `A1_ASK_BUDGET` -> `A3_ASK_POSTCODE` | Tools: `[]` | Latency: `2.57 ms`_

**Visitor:** sw3 3rb

**Assistant:** Great news, Alice! Here are 2 matching available properties in SW3 3RB:
- Apartment 2, Aldermoor Place — 2 bedrooms, £1,230,000
- Apartment 1, Aldermoor Place — 2 bedrooms, £1,330,000

Alice, can I get your phone number?

> _Step: `A3_ASK_POSTCODE` -> `E1_ASK_PHONE` | Tools: `['properties_search']` | Latency: `2.39 ms`_

**Visitor:** 07700 900123

**Assistant:** Alice, can I get your email address?

> _Step: `E1_ASK_PHONE` -> `E2_ASK_EMAIL` | Tools: `[]` | Latency: `2.73 ms`_

**Visitor:** alice@example.com

**Assistant:** Someone will get in touch within 24 hours via phone or email. Is there anything else I can help you with?

> _Step: `E2_ASK_EMAIL` -> `E3_REASSIST` | Tools: `[]` | Latency: `2.28 ms`_

**Visitor:** No, thank you

**Assistant:** Thank you for chatting with us. Goodbye.

> _Step: `E3_REASSIST` -> `ENDED` | Tools: `[]` | Latency: `3.1 ms`_

---
## Final Qualification State
- `qualified`: `True`
- `closing_code`: `GOODBYE`
- `visitor_name`: `Alice`
- `intent`: `buy`
- `listing_category`: `new_build`
- `budget_gbp`: `1500000`
- `postcode`: `SW3 3RB`
- `phone`: `07700 900123`
- `email`: `alice@example.com`
