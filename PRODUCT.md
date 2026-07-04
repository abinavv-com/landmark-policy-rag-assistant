# Product

## Register

product

## Users

Landmark Group retail executives (Lifestyle, Max, Splash, Home Centre, Centrepoint, Babyshop) watching a live scoping-call demo from an AI consulting team. They are evaluating whether to move forward with an engagement, not using this as a daily tool. Their primary skepticism going in is that AI answers are hallucinated or vague. The demo's job is to prove, visibly and in real time, that answers are grounded in actual retrieved documents rather than invented.

## Product Purpose

A Policy & Support RAG Assistant: a customer-service question is typed in, the system retrieves real policy document chunks by embedding similarity, re-ranks them, narrows to the final grounding context, generates an answer, and visually traces that entire path through a graph so the audience can see exactly which document and section the answer came from. Success looks like an executive asking "wait, how do I know that's not made up?" and the interface answering the question itself, live, before anyone has to explain it verbally.

## Brand Personality

Warm, approachable, premium retail. This pulls the assistant's visual language toward the same family as Landmark's other demo prototypes (a warm gold accent, Raleway typography, a premium-but-approachable retail feel) rather than a cold, technical, dark-mode-ops-tool look. The mechanism (the graph, the pipeline stages, the activation scores) stays fully visible and is the trust-building device, but it should read as "a retailer's confident, well-crafted tool" rather than "an engineer's internal dashboard."

## Anti-references

- A generic AI chatbot widget (a bland corner chat bubble) — the graph and pipeline visualization is the point, chat is secondary framing on top of it, not the other way around.
- Generic SaaS dashboard cliches: hero-metric cards, gradient accents, identical stat-tile grids.
- The current implementation's cold, dark, purely technical/ops aesthetic — the underlying graph/pipeline layout structure should be kept as-is, but the visual theme (color, warmth, typography treatment) needs to move toward premium retail rather than staying in "engineer's dashboard" territory.

## Design Principles

- Show the mechanism, don't just claim it. Every visual element (node brightness, flowing lines, stage narrowing) must correspond to a real computed value, never decorative filler standing in for a claim.
- Warmth over coldness. This is a retail brand's tool being shown to retail executives, not an internal engineering console.
- The layout is proven, the skin is not. The graph structure, funnel narrowing, sidebar, and reading pane are validated and should not be restructured — redesign work here is a re-skin, not a rebuild.
- Trust through transparency, not through authority. The interface earns confidence by showing its work at every step, not by asserting correctness.

## Accessibility & Inclusion

Standard WCAG AA contrast target. No specific accessibility requirements beyond that have been raised; this is a live-demo tool for a scoping call, not a daily-use production surface, so the priority is legibility in a conference-room presentation setting over exhaustive accessibility coverage.
