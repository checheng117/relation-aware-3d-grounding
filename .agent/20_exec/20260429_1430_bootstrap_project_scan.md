# Bootstrap Project Scan Report

**Date**: 2026-04-29 14:30
**Status**: COMPLETE
**Repository**: `/srv/shared/ccheng/relation-aware-3d-grounding`

---

## 1. Project Summary

**Project Name**: Latent Conditioned Relation Scoring for 3D Visual Grounding

**Project Type**: Research paper project with codebase for 3D visual grounding

**Status**: Method validation complete (pilot). Full experiments pending - requires headless/remote environment.

**Core Research Question**: Can latent-conditioned relation scoring improve compositional relational inference in 3D referring-expression grounding?

---

## 2. Directory Map

```
relation-aware-3d-grounding/
├── src/rag3d/                      # Core Python package (50+ modules)
│   ├── models/                     # Model implementations
│   │   ├── cover3d_model.py        # COVER-3D wrapper
│   │   ├── cover3d_dense_relation.py
│   │   └── viewpoint_predictor.py
│   ├── relation_reasoner/          # Relation reasoning modules
│   │   ├── viewpoint_conditioned_scorer.py  # PRIMARY METHOD
│   │   ├── latent_relation_mode_scorer.py   # MoE variant
│   │   ├── counterfactual_generator.py
│   │   ├── counterfactual_loss.py
│   │   └── viewpoint_label_generator.py
│   ├── datasets/                   # Data loading (12 modules)
│   ├── diagnostics/                # Analysis tools (8 modules)
│   └── utils/                      # Utilities
├── scripts/                        # Training/evaluation scripts (40+)
│   ├── train_cover3d_counterfactual.py    # Primary training
│   ├── train_cover3d_latent_modes.py      # MoE training
│   ├── evaluate_viewpoint_conditioned_scorer.py
│   └── ... (analysis scripts)
├── configs/                        # Training configs
├── reports/                        # 60+ diagnostic reports (detailed logs)
├── update/                         # 40+ research documentation files
├── course-line/                    # Course project materials
├── writing/                        # Paper drafts
├── outputs/                        # Experimental results
│   ├── phase5_counterfactual/      # Phase 5 pilot results
│   ├── phase6_latent_modes/        # Phase 6 pilot results
│   └── ... (baseline outputs)
├── data/                           # Pre-extracted embeddings, geometry
├── repro/                          # Baseline reproduction
├── tests/                          # Unit tests
├── .agent/                         # Agent state management
└── .claude/                        # Claude project docs
```

---

## 3. Likely Research Goal

**Primary Goal**: Publish a paper demonstrating that latent-conditioned relation scoring improves 3D visual grounding for ambiguous relational expressions.

**Specific Claims Pursued**:
1. **Primary (Supported)**: Latent conditioned architecture provides +2.1% gain over baseline (Phase 4.5 validated)
2. **Secondary (Pilot evidence)**: Counterfactual loss provides +0.12 pp auxiliary improvement (Phase 5 pilot)
3. **Exploratory (Pending)**: Multi-mode latent relation (K=4) may outperform single-mode (Phase 6 pilot shows +0.09 pp)

**Paper Target**: AAAI reviewers, diagnostic/method paper for 3D grounding community

---

## 4. Method Summary

### Primary Architecture: ViewpointConditionedRelationScorer

**Core Design**:
- Augments pairwise relation MLPs with a learnable conditioning variable
- Enables multi-mode relation reasoning
- Viewpoint is one interpretable instance of the latent condition

**Key Insight from Phase 4.5**:
- Architecture gain: +2.1% (E1/E3 vs E0)
- Viewpoint supervision: **NO VALUE** (E1 ≈ E3, difference < 0.1%)
- Gain comes from conditional computation, not supervision signal

### Optional Enhancement: Counterfactual Relation Learning

**Design**:
- Margin ranking loss with counterfactual negatives
- Same-class objects violating specified spatial relation
- Orthogonal to architecture, can be applied to any relation scorer

**Pilot Evidence**: +0.12 pp at 1 epoch (Phase 5)

---

## 5. Existing Evidence

### Level A (Final/Trusted)

| Evidence | Metric | Source |
|----------|--------|--------|
| Scene-disjoint split | Zero overlap verified | `reports/scene_disjoint_split_recovery_results.md` |
| Clean baseline | Acc@1: 30.83% | `outputs/20260420_clean_sorted_vocab_baseline/` |
| Hard subset diagnostics | Same-class: 21.96%, Multi-anchor: 11.90% | `reports/cover3d_phase1_baseline_subset_results.md` |
| Coverage failure | Sparse misses all anchors 33.95% | `reports/cover3d_coverage_diagnostics/` |

### Level B (Controlled)

| Evidence | Metric | Source |
|----------|--------|--------|
| Architecture gain | +2.14% (E1 vs E0) | `update/PHASE4_RESULTS_SUMMARY.md` |
| Supervision not needed | E1 (30.74%) ≈ E3 (30.65%) | `update/PHASE4_5_MECHANISM_DIAGNOSIS.md` |
| Dense-no-cal-v1 | +0.22% overall, +5.3% multi-anchor | `reports/final_diagnostic_master_summary.md` |
| Gain across relations | between: +3.88%, directional: +1.95% | `update/PAPER_CLAIM_REFRAMING.md` |

### Level C (Pilot - Appendix Only)

| Evidence | Metric | Source |
|----------|--------|--------|
| Counterfactual pilot | +0.12 pp (34.42% vs 34.30%) | `update/PHASE5_CONCLUSION_AND_PAPER_NOTES.md` |
| RHN control | FAILED (0% activation) | `update/PHASE5_PILOT_E2_RHN_SAFE_RUN.md` |
| K=4 MoE pilot | +0.09 pp (34.19% vs 34.10%) | `update/PHASE6_PILOT_E1_K4_SAFE_RUN.md` |

---

## 6. Missing Evidence

### Required for Stronger Claims

| Missing Evidence | Why Needed | Status | Blocker |
|-------------------|------------|--------|---------|
| Full training (10 epochs) | Validate pilot results scale | Pending | Requires headless environment |
| Multi-seed validation | Statistical significance | Pending | Compute cost |
| RHN ablation | CF vs Random comparison | FAILED | Missing object metadata in embeddings |
| Multi-anchor improvement | Method addresses multi-anchor | UNSUPPORTED | E1 ≈ E0 ≈ E3 on subset (24.73%) |
| K=4 vs K=1 full training | MoE advantage validation | Pending | 80-epoch training required |

### Data Pipeline Limitations

| Limitation | Impact | Root Cause |
|------------|--------|------------|
| Embeddings-only format | Cannot run RHN ablation | Pre-extracted embeddings lack 3D object metadata |
| No 3D bounding boxes | Cannot compute spatial proximity | Metadata sacrificed for efficient loading |
| No class names | Cannot detect same-class objects | Only class indices stored |

---

## 7. Important Files

### Core Method Code
| File | Purpose |
|------|---------|
| `src/rag3d/relation_reasoner/viewpoint_conditioned_scorer.py` | **Primary method architecture** |
| `src/rag3d/relation_reasoner/latent_relation_mode_scorer.py` | MoE variant (optional) |
| `src/rag3d/relation_reasoner/counterfactual_loss.py` | Counterfactual training objective |
| `src/rag3d/models/cover3d_model.py` | COVER-3D wrapper model |

### Key Documentation
| File | Purpose |
|------|---------|
| `README.md` | Project overview, method status, reproduction guide |
| `reports/final_diagnostic_master_summary.md` | **Master diagnostic summary** |
| `update/PAPER_CLAIM_REFRAMING.md` | Claim comparison and recommendation |
| `update/PHASE5_CONCLUSION_AND_PAPER_NOTES.md` | Phase 5 conclusion, paper positioning |
| `course-line/CLAIM_BOUNDARY.md` | Evidence boundary, what to/not to claim |
| `course-line/FINAL_PROJECT_EVIDENCE_MAP.md` | Evidence level hierarchy |

### Configuration
| File | Purpose |
|------|---------|
| `configs/cover3d_referit_wrapper.yaml` | Main training config |
| `environment.yml` | Conda environment |
| `pyproject.toml` | Package metadata |

---

## 8. Important Scripts

### Training Scripts
| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/train_cover3d_counterfactual.py` | Primary method training | Active |
| `scripts/train_cover3d_latent_modes.py` | MoE training | Active |
| `scripts/train_cover3d_viewpoint.py` | Viewpoint predictor training | Active |
| `scripts/train_cover3d_round1.py` | Baseline training | Reference |

### Evaluation Scripts
| Script | Purpose |
|--------|---------|
| `scripts/evaluate_viewpoint_conditioned_scorer.py` | Method evaluation |
| `scripts/collect_results.py` | Result aggregation |

### Diagnostic Scripts
| Script | Purpose |
|--------|---------|
| `scripts/phase4_5_mechanism_diagnosis.py` | Mechanism analysis |
| `scripts/audit_counterfactual_coverage.py` | CF coverage audit |
| `scripts/sanity_check_latent_mode_training.py` | Training sanity checks |
| `scripts/run_phase7_full.sh` | Full training launcher |

---

## 9. Important Results/Logs

### Primary Results
| Path | Content |
|------|---------|
| `outputs/20260420_clean_sorted_vocab_baseline/` | Baseline evaluation (30.83%) |
| `outputs/phase5_counterfactual/` | CF pilot results |
| `outputs/phase6_latent_modes/` | MoE pilot results |
| `outputs/phase7_full/` | Full training attempt (started) |

### Key Reports
| Path | Content |
|------|---------|
| `reports/final_diagnostic_master_table.csv` | Machine-readable results |
| `reports/final_diagnostic_master_summary.md` | Complete analysis |
| `reports/cover3d_phase1_subset_performance.json` | Subset metrics |

---

## 10. Risks

### Method Risks
| Risk | Severity | Evidence |
|------|----------|----------|
| Counterfactual gain is small (+0.12 pp) | Medium | May not scale to full training |
| RHN ablation impossible | High | Cannot distinguish CF-specific benefit |
| Multi-anchor shows NO improvement | Medium | Core hard subset not addressed |
| Single-seed results only | Medium | Statistical significance unknown |

### Execution Risks
| Risk | Severity | Evidence |
|------|----------|----------|
| Local machine cannot run full training | High | GPU crash history in `update/LOCAL_CRASH_DIAGNOSIS.md` |
| Requires headless/remote environment | High | Pilot works, full fails locally |
| Data pipeline limitations | Medium | Metadata missing from embeddings |

### Paper Risks
| Risk | Severity | Mitigation |
|------|----------|------------|
| Cannot claim "CF > Random" | High | Frame as limitation, honest wording |
| Cannot claim multi-anchor solved | High | Acknowledge as remaining challenge |
| Cannot claim SOTA | Medium | "Improves over reproduced baseline" wording |

---

## 11. Recommended Next 3 Plans

### Plan 1: Complete Full Training Validation (Priority: HIGH)
**Goal**: Validate pilot results scale to 10-epoch full training
**Blocker**: Requires headless/remote environment
**Action**: Configure remote GPU or cloud instance
**Deliverables**: Full E0-matched, E1-CF results (10 epochs)

### Plan 2: Finalize Paper Structure (Priority: MEDIUM)
**Goal**: Write paper with current evidence, honest limitations
**Dependencies**: None (can proceed with pilot evidence)
**Deliverables**: Paper draft with:
- Main claim: Latent conditioned architecture (+2.1%)
- Auxiliary: Counterfactual loss (+0.12 pp pilot)
- Limitations: RHN failed, multi-seed pending

### Plan 3: Metadata Pipeline Enhancement (Priority: LOW)
**Goal**: Enable RHN ablation by re-extracting embeddings with metadata
**Dependencies**: Requires original scene data
**Deliverables**: Updated embeddings with 3D bounding boxes, class names
**Timeline**: Future work, not current priority

---

## Appendix: Key Numbers Summary

### Baseline Performance
- Test Acc@1: 30.83%
- Test Acc@5: 91.87%
- N samples: 4,255 (scene-disjoint)

### Method Performance (Phase 4 Controlled)
- E0 (baseline): 28.60%
- E1 (with supervision): 30.74% (+2.14%)
- E3 (no supervision): 30.65% (+2.04%)

### Pilot Results (1 epoch, batch=8)
- E0-matched (latent-conditioned): 34.30%
- E1-CF (latent-conditioned+cf): 34.42% (+0.12 pp)
- E0-K1 (single mode): 34.10%
- E1-K4 (MoE K=4): 34.19% (+0.09 pp)

### Hard Subset Acc@1
- Same-class clutter: 21.96%
- High clutter: 16.07%
- Multi-anchor: 11.90%
- Multi-anchor (method): 24.73% (no improvement)