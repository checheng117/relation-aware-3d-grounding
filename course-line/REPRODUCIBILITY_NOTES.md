# Reproducibility Notes

This document maps all results to their source files, scripts, and reproduction commands.

---

## Reproduction Overview

| Result Type | Source Location | Script | Command |
|-------------|-----------------|--------|---------|
| Baseline | `outputs/20260420_clean_sorted_vocab_baseline/` | N/A (pre-extracted) | N/A |
| Coverage diagnostics | `reports/cover3d_coverage_diagnostics/` | `scripts/run_cover3d_coverage_diagnostics.py` | See Section 3 |
| Phase 4 results | `outputs/phase4_ablation/` | `scripts/train_cover3d_viewpoint.py` | See Section 4 |
| Phase 5 pilots | `outputs/phase5_counterfactual/` | `scripts/train_cover3d_counterfactual.py` | See Section 5 |
| Phase 6 pilots | `outputs/phase6_latent_modes/` | `scripts/train_cover3d_latent_modes.py` | See Section 6 |

---

## Section 1: Baseline Results

### ReferIt3DNet Baseline (30.83% Acc@1)

**Source File:**
```
outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json
```

**Key Metrics:**
- Acc@1: 30.83%
- Acc@5: 91.87%
- Samples: 4,255 test

**Reproduction:**
Baseline is pre-extracted logits. No training required.

**Verification Command:**
```bash
python -c "import json; d=json.load(open('outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json')); print(f\"Acc@1: {d['acc1']:.2f}%, Acc@5: {d['acc5']:.2f}%\")"
```

**Notes:**
- Uses scene-disjoint split
- Sorted vocabulary baseline
- Logits pre-extracted for efficiency

---

## Section 2: Hard Subset Diagnostics

### Subset Accuracy Table

**Source File:**
```
reports/cover3d_phase1_baseline_subset_results.md
```

**Key Metrics:**
- Overall: 30.83%
- Same-class clutter: 21.96% (2,373 / 4,255)
- High clutter: 16.07% (697 / 4,255)
- Multi-anchor: 11.90% (168 / 4,255)

**Reproduction:**
Subset definitions in `reports/cover3d_phase1_hard_subsets.md`.

**Verification:**
Read subset counts and accuracies from source markdown.

---

## Section 3: Coverage Diagnostics

### Coverage@k Table

**Source File:**
```
reports/cover3d_coverage_diagnostics/coverage_diagnostics_report.md
reports/cover3d_coverage_diagnostics/coverage_summary.json
```

**Key Metrics:**
- Coverage@5 (any anchor): 67.87%
- Coverage@5 (all anchors): 54.20%
- Sparse miss all: 110/324 (33.95%)

**Reproduction Command:**
```bash
python scripts/run_cover3d_coverage_diagnostics.py \
    --manifest data/processed/scene_disjoint/official_scene_disjoint/test_manifest.jsonl \
    --predictions outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json \
    --annotations data/raw/referit3d/annotations/nr3d_annotations.json \
    --geometry-dir data/geometry \
    --output-dir outputs/cover3d_coverage_diagnostics/
```

**Output Files:**
- `outputs/cover3d_coverage_diagnostics/coverage_diagnostics_report.md`
- `outputs/cover3d_coverage_diagnostics/coverage_summary.json`

**Notes:**
- Diagnostic only (no training)
- Requires geometry files for anchor distance computation
- Uses target-centric nearest-neighbor proxy for sparse selection

---

## Section 4: Phase 4 Controlled Results

### E0/E1/E2/E3 Ablation

**Source Files:**
```
outputs/phase4_ablation/dense-v4-hardneg_results.json        (E0)
outputs/phase4_ablation/viewpoint-conditioned_results.json   (E1)
outputs/phase4_ablation/dense-v4-hardneg-extra_results.json  (E2)
outputs/phase4_ablation/viewpoint-conditioned-no-supervision_results.json (E3)
```

**Key Metrics:**
| Experiment | Acc@1 | Gain |
|------------|-------|------|
| E0 | 28.60% | — |
| E1 | 30.74% | +2.14% |
| E2 | 28.74% | +0.14% |
| E3 | 30.65% | +2.04% |

**Reproduction Command (E1 Example):**
```bash
python scripts/train_cover3d_viewpoint.py \
    --variant viewpoint-conditioned \
    --epochs 10 \
    --batch-size 64 \
    --seed 42 \
    --output-dir outputs/phase4_ablation/
```

**Variant Names:**
- `dense-v4-hardneg` → E0
- `viewpoint-conditioned` → E1
- `dense-v4-hardneg-extra` → E2
- `viewpoint-conditioned-no-supervision` → E3

**Output Files:**
- `*_results.json` — Metrics
- `training.log` — Training logs (if saved)

**Notes:**
- All experiments use seed=42
- 10 epochs, batch=64
- Different protocol from baseline (trained model, not pre-extracted logits)

---

## Section 5: Phase 5 Counterfactual Pilots

### E0/E1-CF/E2-RHN Comparison

**Source Files:**
```
outputs/phase5_counterfactual/pilot_E0_bs8_safe/latent-conditioned_results.json
outputs/phase5_counterfactual/pilot_E1_CF_safe/latent-conditioned+cf_results.json
outputs/phase5_counterfactual/pilot_E2_RHN_safe/random-hard-neg_results.json
```

**Key Metrics:**
| Experiment | Val Acc@1 | CF Loss |
|------------|-----------|---------|
| E0-matched | 34.30% (1160/3382) | 0.0000 |
| E1-CF | 34.42% (1164/3382) | 0.2458 |
| E2-RHN | 34.30% (1160/3382) | 0.0000 |

**Reproduction Command (E1-CF Example):**
```bash
python scripts/train_cover3d_counterfactual.py \
    --variant latent-conditioned+cf \
    --epochs 1 \
    --batch-size 8 \
    --seed 42 \
    --train-data outputs/20260420_clean_sorted_vocab_baseline/embeddings/train_embeddings.json \
    --val-data outputs/20260420_clean_sorted_vocab_baseline/embeddings/val_embeddings.json \
    --output-dir outputs/phase5_counterfactual/pilot_E1_CF_safe/
```

**Notes:**
- Pilot: 1 epoch, batch=8
- Uses embedding format (not logits)
- Validation split only
- E2-RHN failed to activate (0/30,447 RHN coverage)

---

## Section 6: Phase 6 Latent-Mode Pilots

### K=1 vs K=4 Comparison

**Source Files:**
```
outputs/phase6_latent_modes/pilot_E0_K1_safe/pilot_e0_k1_safe/training_history.json
outputs/phase6_latent_modes/pilot_E1_K4_safe/pilot_e1_k4_safe/training_history.json
```

**Key Metrics:**
| Experiment | Val Acc@1 | Params |
|------------|-----------|--------|
| K=1 | 34.10% | 328,706 |
| K=4 | 34.19% | 1,216,136 |

**Reproduction Command (K=1 Example):**
```bash
python scripts/train_cover3d_latent_modes.py \
    --num-relation-modes 1 \
    --epochs 1 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --train-data outputs/20260420_clean_sorted_vocab_baseline/embeddings/train_embeddings.json \
    --val-data outputs/20260420_clean_sorted_vocab_baseline/embeddings/val_embeddings.json \
    --name pilot_e0_k1_safe \
    --output-dir outputs/phase6_latent_modes/pilot_E0_K1_safe/
```

**Reproduction Command (K=4 Example):**
```bash
python scripts/train_cover3d_latent_modes.py \
    --num-relation-modes 4 \
    --epochs 1 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --train-data outputs/20260420_clean_sorted_vocab_baseline/embeddings/train_embeddings.json \
    --val-data outputs/20260420_clean_sorted_vocab_baseline/embeddings/val_embeddings.json \
    --name pilot_e1_k4_safe \
    --output-dir outputs/phase6_latent_modes/pilot_E1_K4_safe/
```

**Notes:**
- Pilot: 1 epoch, batch=8
- Uses embedding format
- Validation split only
- K=4 shows +0.09 pp at 1 epoch

---

## Section 7: Figure Sources

### Figure 1: Subset Accuracy

**Source:** `reports/cover3d_phase1/fig1_subset_accuracy.png`

**Copied to:** `writing/course-line/assets/fig1_subset_accuracy.png`

**Reproduction:** Generated by `scripts/generate_cover3d_phase1_figures.py`

---

### Figure 2: Clutter Impact

**Source:** `reports/cover3d_phase1/fig2_clutter_impact.png`

**Copied to:** `writing/course-line/assets/fig2_clutter_impact.png`

---

### Figure 3: Failure Taxonomy

**Source:** `reports/cover3d_phase1/fig5_failure_taxonomy.png`

**Copied to:** `writing/course-line/assets/fig5_failure_taxonomy.png`

---

### Figure 4: Hard vs Easy

**Source:** `reports/cover3d_phase1/fig6_hard_vs_easy.png`

**Copied to:** `writing/course-line/assets/fig6_hard_vs_easy.png`

---

### Figure 5: Coverage Curve (To Generate)

**Source Data:** `reports/cover3d_coverage_diagnostics/coverage_summary.json`

**Script:** Create `scripts/generate_coverage_curve.py`

**Output:** `writing/course-line/assets/coverage_curve.png`

---

### Figure 6: Architecture Diagram (To Create)

**Tool:** draw.io, TikZ, or similar

**Output:** `writing/course-line/assets/architecture_diagram.pdf`

---

## Section 8: Table Sources

### Table 1: Main Results

**Data Sources:**
- Baseline row: `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json`
- Phase 4 rows: `outputs/phase4_ablation/*.json`
- Subset columns: `reports/cover3d_phase1_baseline_subset_results.md`

**Assembly:** Manual (copy numbers to LaTeX table)

---

### Table 2: Coverage Diagnostics

**Data Source:** `reports/cover3d_coverage_diagnostics/coverage_diagnostics_report.md`

**Assembly:** Direct copy from report

---

### Table 3: Ablation by Relation Type

**Data Source:** `update/PHASE4_5_MECHANISM_DIAGNOSIS.md`

**Assembly:** Direct copy

---

## Section 9: Environment Setup

### Requirements

```bash
pip install torch numpy scipy
```

### Data Prerequisites

```
data/processed/scene_disjoint/official_scene_disjoint/test_manifest.jsonl
data/raw/referit3d/annotations/nr3d_annotations.json
data/geometry/  (per-scene geometry files)
```

### Embedding Files (Phase 5/6)

```
outputs/20260420_clean_sorted_vocab_baseline/embeddings/train_embeddings.json
outputs/20260420_clean_sorted_vocab_baseline/embeddings/val_embeddings.json
```

---

## Section 10: Known Reproduction Gaps

| Gap | Impact | Fix |
|-----|--------|-----|
| Phase 4 training logs not saved | Cannot reproduce training curves exactly | Re-run or approximate from results |
| Coverage curve figure not auto-generated | Manual creation needed | Write `scripts/generate_coverage_curve.py` |
| Architecture diagram not auto-generated | Manual creation needed | Use draw.io or TikZ |
| Per-sample predictions not exported | Cannot run significance tests | Add export to evaluation script |
| SAT baseline number conflicts | 28.27% vs 29.17% | Audit source files |

---

## Quick Reference: Commands by Result

| Result | Command |
|--------|---------|
| Coverage diagnostics | `python scripts/run_cover3d_coverage_diagnostics.py --manifest ...` |
| Phase 4 E1 | `python scripts/train_cover3d_viewpoint.py --variant viewpoint-conditioned ...` |
| Phase 5 E1-CF | `python scripts/train_cover3d_counterfactual.py --variant latent-conditioned+cf ...` |
| Phase 6 K=4 | `python scripts/train_cover3d_latent_modes.py --num-relation-modes 4 ...` |

---

## Files Referenced

### Scripts
- `scripts/run_cover3d_coverage_diagnostics.py`
- `scripts/train_cover3d_viewpoint.py`
- `scripts/train_cover3d_counterfactual.py`
- `scripts/train_cover3d_latent_modes.py`
- `scripts/generate_cover3d_phase1_figures.py`

### Outputs
- `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json`
- `outputs/phase4_ablation/*.json`
- `outputs/phase5_counterfactual/pilot_*/`
- `outputs/phase6_latent_modes/pilot_*/`

### Reports
- `reports/cover3d_phase1_baseline_subset_results.md`
- `reports/cover3d_coverage_diagnostics/coverage_diagnostics_report.md`
- `update/PHASE4_RESULTS_SUMMARY.md`
