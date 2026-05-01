# Claim Boundary Document

This document defines what can and cannot be claimed in the course report, why, and how to frame each.

---

## Claim Taxonomy

| Claim Status | Meaning | Course-line treatment |
|--------------|---------|----------------------|
| **Supported** | Evidence strongly supports claim | Use in main text |
| **Qualified** | Evidence supports with caveats | Use with explicit caveats |
| **Unsupported** | Evidence does not support | Do not claim; reframe or omit |
| **Pending** | Insufficient evidence yet | Future work only |

---

## Supported Claims (Main Text OK)

### Claim 1: Baseline Performance on Scene-Disjoint Split

**Claim:** "ReferIt3DNet achieves 30.83% Acc@1 on our scene-disjoint test split (N=4,255)"

| Attribute | Value |
|-----------|-------|
| **Status** | ✅ Supported |
| **Evidence** | `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json` |
| **Evidence Level** | A (Final) |
| **Caveats** | Single seed, but stable across diagnostics |
| **Recommended Wording** | "We reproduce the ReferIt3DNet baseline at 30.83% Acc@1 on our scene-disjoint test split." |

---

### Claim 2: Systematic Hard Subset Failures

**Claim:** "Baseline fails systematically on same-class clutter (21.96%), high clutter (16.07%), and multi-anchor (11.90%) subsets"

| Attribute | Value |
|-----------|-------|
| **Status** | ✅ Supported |
| **Evidence** | `reports/cover3d_phase1_baseline_subset_results.md` |
| **Evidence Level** | A (Final) |
| **Caveats** | Deterministic subset analysis |
| **Recommended Wording** | "Diagnostic analysis reveals systematic failures: accuracy drops to 12-22% on hard subsets." |

---

### Claim 3: Coverage Failure

**Claim:** "Sparse top-5 anchor selection misses all annotated anchors in 33.95% (110/324) of baseline-wrong, anchor-evaluable samples"

| Attribute | Value |
|-----------|-------|
| **Status** | ✅ Supported |
| **Evidence** | `reports/cover3d_coverage_diagnostics/coverage_diagnostics_report.md` |
| **Evidence Level** | A (Final) |
| **Caveats** | Diagnostic only — does not prove dense reranking improves accuracy |
| **Recommended Wording** | "Coverage analysis shows sparse selection misses all anchors in 34% of baseline errors, motivating dense candidate coverage." |

---

### Claim 4: Architecture Benefit (Phase 4)

**Claim:** "Viewpoint-conditioned architecture provides +2.14% gain over baseline in controlled setting"

| Attribute | Value |
|-----------|-------|
| **Status** | ✅ Supported |
| **Evidence** | `update/PHASE4_RESULTS_SUMMARY.md`, `outputs/phase4_ablation/*.json` |
| **Evidence Level** | B (Controlled) |
| **Caveats** | Single seed, different protocol from main baseline |
| **Recommended Wording** | "Controlled experiments show the viewpoint-conditioned architecture improves accuracy by +2.14% over a matched baseline." |

---

### Claim 5: Architecture vs Supervision

**Claim:** "Gain comes from architecture, NOT viewpoint supervision (E3 without supervision matches E1 with supervision)"

| Attribute | Value |
|-----------|-------|
| **Status** | ✅ Supported |
| **Evidence** | `update/PHASE4_5_MECHANISM_DIAGNOSIS.md` |
| **Evidence Level** | B (Controlled) |
| **Caveats** | Single seed |
| **Recommended Wording** | "Ablation shows explicit viewpoint supervision adds no value — the architecture itself provides the gain." |

---

## Qualified Claims (Use with Caveats)

### Claim 6: Dense-no-cal-v1 Improvement

**Claim:** "Dense-no-cal-v1 shows +0.22% overall gain, +5.3% on multi-anchor subset"

| Attribute | Value |
|-----------|-------|
| **Status** | ⚠️ Qualified |
| **Evidence** | `reports/final_diagnostic_master_summary.md` |
| **Evidence Level** | B (Controlled) |
| **Caveats** | 10 epochs, single seed, modest overall gain |
| **Recommended Wording** | "A simple dense relation scorer shows +0.22% overall gain and +5.3% on multi-anchor cases (10 epochs, single seed)." |

---

### Claim 7: Gain Across Relation Types

**Claim:** "Improvement is general across relation types (between: +3.88%, directional: +1.95%, support: +2.30%, attribute: +1.67%)"

| Attribute | Value |
|-----------|-------|
| **Status** | ⚠️ Qualified |
| **Evidence** | `update/PHASE4_5_MECHANISM_DIAGNOSIS.md` |
| **Evidence Level** | B (Controlled) |
| **Caveats** | From Phase 4 protocol, may not generalize |
| **Recommended Wording** | "Per-relation analysis shows gains across all types, suggesting the architecture provides general improvement." |

---

## Unsupported Claims (Do NOT Claim)

### Claim 8: "K=4 MoE Improves Over K=1"

**Claim:** "Latent relation modes (K=4) outperform single-mode (K=1)"

| Attribute | Value |
|-----------|-------|
| **Status** | ❌ Unsupported |
| **Evidence** | `update/PHASE6_PILOT_E1_K4_SAFE_RUN.md` |
| **Evidence Level** | C (Pilot) |
| **Why Unsupported** | Only 1 epoch (+0.09 pp signal too small); full 80-epoch training pending |
| **Reframed Wording** | "K=4 shows early pilot signal (+0.09 pp at 1 epoch); full training validation is ongoing." |
| **Placement** | Future work Section 6, NOT main results |

---

### Claim 9: "Counterfactual > Random Hard Negatives"

**Claim:** "Counterfactual negatives are superior to random hard negatives"

| Attribute | Value |
|-----------|-------|
| **Status** | ❌ Unsupported |
| **Evidence** | `update/PHASE5_PILOT_E2_RHN_SAFE_RUN.md` |
| **Evidence Level** | C (Pilot) |
| **Why Unsupported** | RHN never activated (0/30,447 coverage); E2-RHN = E0-matched (both 34.30%) |
| **Reframed Wording** | "The RHN control did not activate due to missing object metadata in pre-extracted embeddings. We cannot distinguish counterfactual-specific benefits from general hard negative effects." |
| **Placement** | Limitations Section 5.1, NOT main claims |

---

### Claim 10: "Viewpoint Supervision Helps"

**Claim:** "Explicit viewpoint supervision improves 3D grounding"

| Attribute | Value |
|-----------|-------|
| **Status** | ❌ Unsupported |
| **Evidence** | `update/PHASE4_RESULTS_SUMMARY.md` |
| **Evidence Level** | B (Controlled) |
| **Why Unsupported** | E3 (no supervision) achieves 30.65% vs E1 (with supervision) 30.74% — effectively identical |
| **Reframed Wording** | "The viewpoint-conditioned architecture improves grounding, but explicit viewpoint supervision is not necessary." |
| **Placement** | Reframe as architecture claim (Claim 5) |

---

### Claim 11: "Method Solves Multi-Anchor Grounding"

**Claim:** "Our method addresses multi-anchor expression grounding"

| Attribute | Value |
|-----------|-------|
| **Status** | ❌ Unsupported |
| **Evidence** | `update/PHASE4_RESULTS_SUMMARY.md` |
| **Evidence Level** | B (Controlled) |
| **Why Unsupported** | Multi-anchor subset: E0 24.73%, E1 24.73%, E3 23.66% — NO improvement |
| **Reframed Wording** | "Multi-anchor grounding remains challenging; our method shows no improvement on this subset." |
| **Placement** | Limitations Section 5.1 |

---

### Claim 12: "SOTA on ReferIt3D"

**Claim:** "Our method achieves state-of-the-art on ReferIt3D"

| Attribute | Value |
|-----------|-------|
| **Status** | ❌ Unsupported |
| **Evidence** | N/A |
| **Evidence Level** | N/A |
| **Why Unsupported** | (1) Only single-seed results, (2) Scene-disjoint split not comparable to literature, (3) SAT comparison has number conflicts |
| **Reframed Wording** | "Our method improves over our reproduced baseline on the scene-disjoint split. Comparison to literature is complicated by potential split differences." |
| **Placement** | Do not claim SOTA |

---

## Pending Claims (Future Work Only)

### Claim 13: "Latent-Mode Full Training Validates MoE Advantage"

**Claim:** "Full 80-epoch training shows K=4 MoE significantly outperforms K=1"

| Attribute | Value |
|-----------|-------|
| **Status** | ⏳ Pending |
| **Evidence** | Not yet available |
| **Expected** | `outputs/latent_mode/e0_k1_baseline/`, `outputs/latent_mode/e1_k4_primary/` |
| **Placement** | Future work Section 6 |

---

### Claim 14: "Multi-Seed Validation Shows Stability"

**Claim:** "Results are stable across seeds (±X%)"

| Attribute | Value |
|-----------|-------|
| **Status** | ⏳ Pending |
| **Evidence** | Not yet available |
| **Placement** | Future work Section 6 |

---

### Claim 15: "Metadata-Preserving Pipeline Enables RHN"

**Claim:** "With full object metadata, RHN ablation becomes feasible"

| Attribute | Value |
|-----------|-------|
| **Status** | ⏳ Pending |
| **Evidence** | `scripts/migrate_embeddings_with_metadata.py` (partial) |
| **Placement** | Future work Section 6 |

---

## Summary: What TO Say vs What NOT To Say

| Instead of Saying... | Say This... |
|----------------------|-------------|
| "K=4 MoE is better than K=1" | "K=4 shows early pilot signal; full training pending" |
| "Counterfactual > Random" | "RHN control inconclusive due to data format" |
| "Viewpoint supervision helps" | "Architecture helps; supervision not necessary" |
| "We solve multi-anchor" | "Multi-anchor remains challenging" |
| "SOTA on ReferIt3D" | "Improves over our baseline on scene-disjoint split" |
| "Our method is novel" | "We apply latent conditioned scoring to 3D grounding" |
| "We prove X" | "Our experiments suggest X" |

---

## Evidence Boundary Figure (for Report)

```
Can Claim                    Can Claim with Caveats         Cannot Claim (Yet)
─────────────────            ────────────────────         ──────────────────
• Baseline 30.83%            • Dense-no-cal-v1 +0.22%     • K=4 > K=1 (pilot only)
• Hard subset failures       • Phase 4 gain +2.1%         • CF > RHN (RHN failed)
• Coverage gap 34%           • Gain across relations      • Viewpoint supervision helps
• Architecture > Param count •                            • Multi-anchor solved
• Supervision NOT needed     •                            • SOTA claim
```

---

## How to Frame Limitations as Strengths

| Limitation | Reframe for Course Report |
|------------|--------------------------|
| Pilot results only | "Transparent evidence hierarchy — pilots clearly labeled" |
| Single-seed results | "Honest about stability limitations; multi-seed is future work" |
| RHN failed | "Identified root cause (metadata format); pipeline improvement direction" |
| Multi-anchor no gain | "Clear diagnosis of remaining challenge; not over-claiming" |
| Protocol differences | "Explicitly documented; tables footnote incompatibilities" |

**Key principle:** Instructors value intellectual honesty over over-selling. A well-documented limitation is better than a misleading claim.

---

## Files Referenced

- `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json`
- `outputs/phase4_ablation/*.json`
- `outputs/phase5_counterfactual/pilot_*/`
- `outputs/phase6_latent_modes/pilot_*/`
- `reports/cover3d_phase1_baseline_subset_results.md`
- `reports/cover3d_coverage_diagnostics/coverage_diagnostics_report.md`
- `update/PHASE4_RESULTS_SUMMARY.md`
- `update/PHASE4_5_MECHANISM_DIAGNOSIS.md`
- `update/PHASE5_CONCLUSION_AND_PAPER_NOTES.md`
- `update/PHASE6_PILOT_E1_K4_SAFE_RUN.md`
