# Hybrid router phase — summary

## 1. What was implemented

- **README** reframing (complementarity hypothesis, \(\alpha(x)\) fusion, ceiling analyses, phased roadmap) plus `reports/readme_reframing_note.md`.
- **Audit** `reports/hybrid_router_audit.md` (artifact locations, per-sample signals, protocol).
- **Script** `scripts/run_hybrid_router_phase.py`: frozen B/C forward passes over train/val manifests, **oracle shortlist** (gold forced into coarse top-K, rerank head unchanged), **oracle geometry** slices (primary tag + median fallback when degenerate), **oracle branch** (discrete OR of B/C correctness), **lightweight fusion router** (linear \(\alpha(x)\) with mixture NLL on train, probability-mixture prediction on val).
- **Output bundle** `outputs/20260327_224200_hybrid_router_phase/` (JSON, CSV, interpretation markdowns, `hybrid_router_main_figure.png`, `hybrid_router_log.txt`, `repro_commands.sh`, `report_bundle/README.md`).

## 2. What was reused from prior runs

- Checkpoints: `outputs/checkpoints_nr3d_geom_first/raw_relation_last.pt` (B), `relation_aware_last.pt` (C).
- Two-stage stack: `outputs/checkpoints_stage1/coarse_geom_recall_last.pt` + `outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt`.
- Manifests: `data/processed/train_manifest.jsonl` (router training features), `val_manifest.jsonl` (held-out metrics).
- Library: `eval_two_stage_bottleneck`, `load_two_stage_model`, stratification tags, structured parser cache path under `data/parser_cache/hybrid_phase/structured`.

## 3. Is B/C complementarity real or weak? (this val split)

On **val** (n=156), **oracle branch** Acc@1 ≈ **0.058** vs max(B, C) ≈ **0.032** — a **small but non-zero** OR-gap (~**0.026**), so **some** samples are solved by exactly one branch. This is **weak complementarity** at the current accuracy level (both lines are noisy), not a large disjoint support.

**Regimes:** Oracle branch is higher on **candidate_load_high** (n=39) than on **low** in this run; `same_class_clutter` shows n=156 because, with the current heuristic, **every** val sample is tagged as clutter—treat that slice as **non-informative** until the tag is tightened or a subset is redefined.

## 4. Do the oracle analyses justify continuing?

| Analysis | Verdict |
|----------|---------|
| **Oracle shortlist** | **Yes, for two-stage work.** Natural shortlist recall ≈ **0.31**; **oracle shortlist** Acc@1 ≈ **0.47** with the **same** reranker. Retrieval misses a large share of targets; even when the target is in K, conditional rerank Acc@1 stays low (~**0.10**). So **both** recall and rerank quality matter. |
| **Oracle geometry** | **Inconclusive here.** No val samples fell in **geometry_high_fallback**; `geometry_fallback_fraction` is **degenerate** (median split collapses). The run **cannot** test “C benefits more from clean geometry” on this slice—need manifests with varied `geometry_quality` / fallback rates or a larger val set. |
| **Oracle branch** | **Weak support for a router** at current accuracy: OR-gap is small; fusion captures only part of it (below). |

## 5. Does the lightweight router beat standalone B/C?

**Deterministic settings** (see script): val Acc@1 **B ≈ 0.032**, **C ≈ 0.026**, **fusion ≈ 0.045**. So fusion **edges both** baselines by a **small** margin (~**+0.013** over B). It **does not** approach the oracle branch (~**0.058**): roughly **half** of the OR-headroom is recovered with this feature set and linear \(\alpha\).

**Feature magnitudes** (mean |weight|, coarse proxy): large on **margins**, **confidences**, **candidate_load_high**, and **relation_density** / **parser_conf** in this run—consistent with routing on **uncertainty and load** rather than a single geometry flag.

## 6. Next best step

1. **Shortlist / stage-1** — oracle shortlist shows the largest structural headroom; improving **Recall@K** and **rerank given in-K** is higher leverage than pushing a router on ~3% Acc@1 baselines.
2. **Router** — revisit after stronger B/C or richer features (e.g. calibrated scores, explicit shortlist rank); current evidence says **small** gain, not a stop condition by itself.
3. **Geometry-aware modeling** — keep on the roadmap, but **this val manifest** did not expose a high-fallback slice; validate on data with real spread in OBB coverage.
4. **Stop?** — **No full stop**: complementarity and shortlist ceilings are **real but modest**; prioritize **shortlist-aligned training** next, then a **second router pass** on stronger frozen logits.
