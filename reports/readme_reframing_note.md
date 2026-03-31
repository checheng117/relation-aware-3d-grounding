# README reframing note

## Sections added or changed

- **Main findings**: Added a short paragraph stating that multi-seed B vs C can be seed- and regime-sensitive, and that the repo should not claim uniform C-over-B dominance.
- **Updated research hypothesis and next steps** (new): Reframes the core question around *stable gains where structure should help*; states complementarity of raw-text vs structured paths; defines the gated fusion \(s_{\mathrm{final}} = \alpha(x) s_B + (1-\alpha(x)) s_C\); lists candidate router features; summarizes hybrid motivation; outlines three improvement directions (hybrid router, shortlist-aware training, geometry-aware modeling); points to `outputs/<timestamp>_hybrid_router_phase/` and `reports/hybrid_router_phase_summary.md`.
- **Ceiling analyses before major refactoring** (new): Describes oracle shortlist, oracle geometry, and oracle branch selector and what each is meant to bound.
- **Staged roadmap** (new): Phases A–D (ceiling → lightweight router → shortlist training → geometry-aware modeling), with A–B as script-backed and C–D as proposed.
- **Future work**: Prefixed with executing C–D when evidence supports it; otherwise unchanged in spirit.

## Why this reframing aligns with current evidence

Official and multi-seed runs show that **B vs C is not a single stable ordering** across seeds and slices; overclaiming “structured always wins” would misrepresent the diagnostics. The new framing keeps the **original motivation** (structured, inspectable grounding) but ties claims to **regimes and bottlenecks** (shortlist, geometry, ambiguity) and to **complementarity**, which is exactly what oracle branch and hybrid-router experiments are designed to test. The README now separates **what the repo has shown** from **what is proposed next**, in line with a research-oriented, non-hype tone.
