# Ashcombe & Rowe Estates — Lead Qualification

Domain language for the Ashcombe & Rowe Estates lead qualification chatbot across buyer, seller, and mid-flow Q&A journeys.

## Language

**Visitor**:
A person interacting with the chatbot before their inquiry is qualified or handed off.
_Avoid_: User, client, customer

**Lead**:
A qualified Visitor whose phone number and email address have been captured for follow-up by the agency team.
_Avoid_: Prospect, contact, qualified user

**Session**:
A stateful multi-turn conversation with a single Visitor, tracked by a unique session identifier from initial greeting through closing or reassist.
_Avoid_: Thread, chat, run

**Turn**:
One Visitor message together with the single composed reply, updated session state, and recorded telemetry produced in response.
_Avoid_: Request, step execution

**Journey Lane**:
One of the five deterministic qualification paths in the agency flowchart: New-Build Buyer (`A`), Resale Buyer (`B`), Specific Property Buyer (`C`), Seller (`D`), and Contact Capture / Reassist (`E`).
_Avoid_: Workflow branch, pipeline

**Mid-Flow Interrupt**:
A Visitor Turn that asks an FAQ question, asks about a property, goes off-topic, or attempts a rule override while keeping the active workflow step ready to resume.
_Avoid_: Digression, exception

**Mixed Turn**:
A Visitor Turn that both answers the active workflow step and asks a side FAQ or property question in the same message.
_Avoid_: Dual intent

**Mind Change**:
An explicit Visitor Turn mid-conversation that switches Journey Lane or buying preference (for example switching from buying to selling, or from new-build to resale).
_Avoid_: Reset, override

**Rule Override**:
A Visitor attempt to bypass agency qualification policy (such as the £1 million new-build floor or seller postcode coverage) or manipulate system instructions.
_Avoid_: Jailbreak, policy error
