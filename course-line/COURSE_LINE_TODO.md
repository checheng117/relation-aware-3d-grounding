# Course-Line TODO List: Path to 95+

This document lists all enhancement tasks for the CSC6133 Final Project, prioritized by impact on course score.

---

## Priority Legend

| Priority | Meaning | Action |
|----------|---------|--------|
| **P0** | Must-have for 95+ | Do immediately |
| **P1** | Should-have for 95+ | Do after P0 |
| **P2** | Nice-to-have | Do if time permits |

---

## P0: Must-Have for 95+

### P0-1: Unified Protocol Main Table

| Attribute | Value |
|-----------|-------|
| **Goal** | Create one authoritative table with all methods under same protocol |
| **Actions** | 1. Extract baseline (30.83%) from `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json`<br>2. Extract Phase 4 E0/E1/E3 from `outputs/phase4_ablation/*.json`<br>3. Format as single table with consistent metric definitions<br>4. Add subset rows (same-class, high-clutter, multi-anchor) |
| **Deliverable** | `writing/course-line/tables/main_results_table.tex` or CSV |
| **Risk** | Numbers come from different protocols (logits vs embeddings) — must add footnote |
| **Score Impact** | +10 points (shows experimental rigor) |
| **Effort** | 2-3 hours |

---

### P0-2: Evidence Level Labels in Report

| Attribute | Value |
|-----------|-------|
| **Goal** | Explicitly label each result as Final/Controlled/Pilot in report text |
| **Actions** | 1. Add footnote markers to tables (e.g., `^F` for Final, `^C` for Controlled, `^P` for Pilot)<br>2. Add legend explaining levels<br>3. Ensure pilot results are NOT in main table |
| **Deliverable** | Updated `writing/course-line/report.tex` with footnotes |
| **Risk** | None — this is labeling existing results |
| **Score Impact** | +5 points (shows research maturity) |
| **Effort** | 1 hour |

---

### P0-3: Qualitative Case Studies (2 Cases)

| Attribute | Value |
|-----------|-------|
| **Goal** | Add one success case and one failure case with visualization |
| **Actions** | 1. Review `reports/cover3d_phase1/` for existing case visualizations<br>2. Select one clear success (baseline correct → method correct)<br>3. Select one clear failure (baseline wrong → method still wrong)<br>4. Create figure with utterance, scene snapshot, anchor/target highlights<br>5. Add 1-paragraph analysis per case |
| **Deliverable** | `writing/course-line/assets/case_success.png`, `case_failure.png` + report Section 4.4 |
| **Risk** | May need to generate new visualizations if existing ones are insufficient |
| **Score Impact** | +10 points (makes report memorable, shows deep understanding) |
| **Effort** | 3-4 hours |

---

### P0-4: Reproducibility Notes

| Attribute | Value |
|-----------|-------|
| **Goal** | Document exactly how to reproduce each table/figure |
| **Actions** | 1. Map each table to source file and script<br>2. Map each figure to source file and script<br>3. Write exact commands (copy-paste runnable)<br>4. Note any manual steps (e.g., LaTeX compilation) |
| **Deliverable** | `course-line/REPRODUCIBILITY_NOTES.md` (this file will be linked from report) |
| **Risk** | Some results may have incomplete provenance — mark as "partial" if needed |
| **Score Impact** | +5 points (professional package) |
| **Effort** | 2 hours |

---

### P0-5: Presentation Slides (10-15 min)

| Attribute | Value |
|-----------|-------|
| **Goal** | Create polished presentation following `PRESENTATION_PLAN.md` |
| **Actions** | 1. Follow slide-by-slide plan in `PRESENTATION_PLAN.md`<br>2. Use existing figures from `writing/course-line/assets/`<br>3. Keep one core message per slide<br>4. Include backup slides for pilot results |
| **Deliverable** | `writing/course-line/presentation/slides.pdf` |
| **Risk** | Time constraint — focus on main story first, backup slides if time |
| **Score Impact** | +15 points (presentation is major component) |
| **Effort** | 4-6 hours |

---

## P1: Should-Have for 95+

### P1-1: Hard Subset Extended Table

| Attribute | Value |
|-----------|-------|
| **Goal** | Add detailed breakdown of all hard subsets |
| **Actions** | 1. Extract all subset results from `reports/cover3d_phase1_baseline_subset_results.md`<br>2. Add Phase 4 E0/E1 breakdown by subset<br>3. Format as extended table (appendix) |
| **Deliverable** | `writing/course-line/tables/extended_subsets_table.tex` |
| **Risk** | Phase 4 subset breakdown may be incomplete — use what's available |
| **Score Impact** | +5 points (shows thoroughness) |
| **Effort** | 2 hours |

---

### P1-2: Training Curves Figure

| Attribute | Value |
|-----------|-------|
| **Goal** | Show training dynamics for Phase 4 E0/E1 |
| **Actions** | 1. Extract training logs from `outputs/phase4_ablation/*.log`<br>2. Plot loss curves and accuracy curves<br>3. Label convergence behavior |
| **Deliverable** | `writing/course-line/assets/training_curves.png` |
| **Risk** | Logs may not have per-epoch metrics — may need to re-run or approximate |
| **Score Impact** | +3 points (standard deep learning paper element) |
| **Effort** | 2-3 hours |

---

### P1-3: Architecture Diagram

| Attribute | Value |
|-----------|-------|
| **Goal** | Clear visual of latent-conditioned relation scorer |
| **Actions** | 1. Review `writing/paper-line/main_draft.tex` for existing diagram<br>2. Adapt for course audience (more explanatory)<br>3. Show: object embeddings → relation scorer → viewpoint conditioning → output |
| **Deliverable** | `writing/course-line/assets/architecture_diagram.pdf` or `.png` |
| **Risk** | May need graphics tool (draw.io, TikZ) — keep simple |
| **Score Impact** | +5 points (clarifies method, professional look) |
| **Effort** | 3-4 hours |

---

### P1-4: Failure Taxonomy Discussion

| Attribute | Value |
|-----------|-------|
| **Goal** | Systematic categorization of remaining failure modes |
| **Actions** | 1. Use `reports/cover3d_phase1/fig5_failure_taxonomy.png`<br>2. Write 1-2 paragraphs categorizing failures<br>3. Link to limitations section |
| **Deliverable** | Report Section 5.2 (Failure Taxonomy) |
| **Risk** | None — uses existing figure |
| **Score Impact** | +3 points (shows critical thinking) |
| **Effort** | 1-2 hours |

---

### P1-5: Appendix with Extended Materials

| Attribute | Value |
|-----------|-------|
| **Goal** | Collect all supplementary material in one place |
| **Actions** | 1. Create `writing/course-line/appendix/` directory<br>2. Move extended tables, additional figures, training curves here<br>3. Add appendix index |
| **Deliverable** | `writing/course-line/appendix/` with index document |
| **Risk** | None — organizational task |
| **Score Impact** | +3 points (professional packaging) |
| **Effort** | 1 hour |

---

### P1-6: Copy Missing Figures from Phase 1

| Attribute | Value |
|-----------|-------|
| **Goal** | Copy `fig3_scene_size.png` and `fig4_relation_type.png` to assets |
| **Actions** | 1. Copy `reports/cover3d_phase1/fig3_scene_size.png` → `writing/course-line/assets/`<br>2. Copy `reports/cover3d_phase1/fig4_relation_type.png` → `writing/course-line/assets/`<br>3. Add captions and analysis text |
| **Deliverable** | `writing/course-line/assets/fig3_scene_size.png`, `fig4_relation_type.png` |
| **Risk** | None — simple copy |
| **Score Impact** | +2 points (additional diagnostic evidence) |
| **Effort** | 30 min |

---

## P2: Nice-to-Have

### P2-1: Multi-Seed Results

| Attribute | Value |
|-----------|-------|
| **Goal** | Show stability across seeds (42, 123, 456) |
| **Actions** | 1. Re-run Phase 4 E0/E1 with seeds 123 and 456<br>2. Report mean ± std dev |
| **Deliverable** | Updated main table with error bars |
| **Risk** | GPU cost (~4-6 hours per seed); may not finish in time |
| **Score Impact** | +3 points (shows statistical awareness) |
| **Effort** | 12+ hours GPU time + 2 hours analysis |

---

### P2-2: Statistical Significance Tests

| Attribute | Value |
|-----------|-------|
| **Goal** | Paired significance test between baseline and method |
| **Actions** | 1. Extract per-sample predictions from baseline and E1<br>2. Run McNemar's test or paired bootstrap<br>3. Report p-value |
| **Deliverable** | Table footnote: "p < 0.05 by McNemar's test" |
| **Risk** | Requires per-sample predictions; may need additional extraction |
| **Score Impact** | +2 points (sophisticated touch) |
| **Effort** | 2-3 hours |

---

### P2-3: Additional Baseline (SAT)

| Attribute | Value |
|-----------|-------|
| **Goal** | Include SAT baseline for completeness |
| **Actions** | 1. Check `outputs/sat_baseline/20260414_run/eval_test_results.json`<br>2. Verify number conflicts with `reports/final_diagnostic_master_summary.md`<br>3. Add to table with caveat if needed |
| **Deliverable** | SAT row in extended table |
| **Risk** | Number conflicts (28.27% vs 29.17%) — must resolve or footnote |
| **Score Impact** | +2 points (thoroughness) |
| **Effort** | 1-2 hours if numbers are clean |

---

### P2-4: Counterfactual Full Training

| Attribute | Value |
|-----------|-------|
| **Goal** | Validate if +0.12 pp pilot gain scales to full training |
| **Actions** | 1. Run `scripts/train_cover3d_counterfactual.py --variant latent-conditioned+cf --epochs 10`<br>2. Compare to E0 baseline at 10 epochs |
| **Deliverable** | Updated pilot section with full results |
| **Risk** | GPU cost (~4-6 hours); gain may disappear |
| **Score Impact** | +2 points (completes counterfactual story) |
| **Effort** | 6 hours GPU + 1 hour analysis |

---

### P2-5: Latent-Mode Full Training (K=1 vs K=4)

| Attribute | Value |
|-----------|-------|
| **Goal** | Validate if K=4 MoE shows larger advantage at 80 epochs |
| **Actions** | 1. Run `scripts/train_cover3d_latent_modes.py --num-relation-modes 1 --epochs 80` (E0-K1)<br>2. Run `scripts/train_cover3d_latent_modes.py --num-relation-modes 4 --epochs 80` (E1-K4)<br>3. Compare final test accuracy |
| **Deliverable** | New main table row for K=1 and K=4 |
| **Risk** | GPU cost (~10-14 hours total); may show no advantage |
| **Score Impact** | +5 points (validates core method extension) |
| **Effort** | 14 hours GPU + 2 hours analysis |

---

## Task Summary by Priority

| Priority | Count | Total Effort | Score Impact |
|----------|-------|--------------|--------------|
| **P0** | 5 tasks | 12-16 hours | +45 points |
| **P1** | 5 tasks | 9-12 hours | +19 points |
| **P2** | 5 tasks | 20+ hours (mostly GPU) | +14 points |

---

## Recommended Execution Order

1. **Week 1**: P0-1, P0-2, P0-3, P0-4 (core report enhancements)
2. **Week 2**: P0-5, P1-1, P1-2, P1-3 (presentation + appendix)
3. **Week 3**: P1-4, P1-5, P2-1 or P2-5 (polish + optional experiments)

---

## Evidence Enhancement Opportunities

The following repository materials can be enhanced for course-line:

| Existing Material | Enhancement | Placement |
|-------------------|-------------|-----------|
| `reports/cover3d_phase1/fig1_subset_accuracy.png` | Add method comparison (E0/E1) | Report Figure 2 |
| `reports/cover3d_coverage_diagnostics/` | Add coverage curve figure | Report Figure 1 |
| `update/PHASE4_RESULTS_SUMMARY.md` | Convert to academic table | Report Table 3 |
| `reports/cover3d_phase1/fig5_failure_taxonomy.png` | Add discussion text | Report Section 5.2 |
| `reports/cover3d_phase1/fig3_scene_size.png` | Copy to assets, add analysis | Appendix Figure A2 |
| `reports/cover3d_phase1/fig4_relation_type.png` | Copy to assets, add analysis | Appendix Figure A3 |

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| GPU runs fail or hang | Use safe-run protocols from `update/` docs; monitor thermals |
| Number conflicts persist | Add footnotes explaining protocol differences |
| Time runs short | Complete P0 first; P1/P2 are optional for 95+ |
| Visualizations insufficient | Use simple diagrams; prioritize clarity over polish |

---

## Files Referenced

- `outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_results.json`
- `outputs/phase4_ablation/*.json`
- `reports/cover3d_phase1/`
- `reports/cover3d_phase1_baseline_subset_results.md`
- `update/PHASE4_RESULTS_SUMMARY.md`
- `writing/course-line/assets/`
