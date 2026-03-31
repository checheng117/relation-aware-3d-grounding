# Shortlist + rerank upgrade — summary

## 1. What was implemented

- **Audit / plan:** `reports/shortlist_rerank_upgrade_plan.md`
- **Hard-negative design:** `reports/shortlist_hard_negative_design.md`
- **Curriculum:** documented as not run (`reports/shortlist_curriculum_note.md`)
- **Core code:** `TwoStageCoarseRerankModel.forward(..., inject_gold_in_shortlist=...)`; `shortlist_train_inject_gold` + `fine_init_checkpoint` in `train_two_stage_rerank.py`; `--init-checkpoint` + `val_coarse_recall_ks` coarse training; `TrainingConfig.val_coarse_recall_ks` + val logging in `runner.py`; `two_stage_rerank_metrics.py` (Acc@1/5, MRR, conditional in-K, load buckets)
- **Orchestrator:** `scripts/run_shortlist_rerank_upgrade_phase.py` → `outputs/20260327_224830_shortlist_rerank_upgrade/` (example run; your timestamp may differ)
- **README:** minimal roadmap update + `reports/readme_shortlist_rerank_note.md`

## 2. Prior assets reused

- Baseline coarse `outputs/checkpoints_stage1/coarse_geom_recall_last.pt`
- Baseline two-stage `outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt` (fine init + reference pipeline)
- NR3D processed `train_manifest.jsonl` / `val_manifest.jsonl`
- Existing hinge / same-class / spatial losses for coarse finetune

## 3. Reranker headroom (oracle-shortlist eval)

On the **example val** run (n=156), **oracle shortlist** Acc@1 is on the order of **0.40–0.45** while **natural** pipeline Acc@1 stays in the **few-percent** range — a large gap, so **retrieval (Recall@K)** remains a dominant term. Under oracle shortlist, **~55–60%** of accuracy potential is still left on the table (1 − oracle Acc@1), so the **rerank head is not solved** either: conditional Acc@1 given gold ∈ K on **natural** shortlist stays **moderate at best** in the same run.

**Oracle-protocol rerank training** (always inject gold during train) can **raise oracle-eval Acc@1 slightly** vs baseline in short training; **natural-protocol** training can help **natural** Acc@1 somewhat — gains are **small** on this budget and noisy.

## 4. Did shortlist-aware retrieval help?

- **Recall@10** on val moved from **~0.31 → ~0.34** in the re-eval pass after **2-epoch** coarse finetune (see `shortlist_retrieval_results.json`); **Recall@5** and **Recall@40** did **not** uniformly improve — finetune is **thin** and can trade off ranks.
- **Val JSONL** logs per-epoch `val_coarse_recall@K` and **same_class_clutter** / **candidate_load::high** slices for checkpoint auditing.

## 5. Combined pipeline vs current

On the **same short run**, **B_natural_protocol** reached **~0.032** natural Acc@1 vs baseline **~0.045** (stochastic — your numbers may differ). **Upgraded coarse + refit rerank (C/D)** did **not** clearly beat the **baseline reference** on **natural** Acc@1 in this 2-epoch experiment; oracle-eval **dropped** vs baseline when paired with the new coarse, suggesting **coarse–rerank mismatch** or **under-training** of the fine head on the new shortlist distribution.

**Interpretation:** longer training and/or **joint** selection of coarse + rerank checkpoints (e.g. maximize val natural Acc@1, not only coarse recall) is needed before claiming a win.

## 6. Next best step

1. **More rerank epochs** on a **fixed** upgraded coarse, with **early stopping** on **natural** two-stage val Acc@1 (not only coarse recall).
2. **Deeper rerank** (capacity / K) only if oracle-shortlist Acc@1 remains the main ceiling after (1).
3. **Curriculum / entity manifests** when available — see `shortlist_curriculum_note.md`.
4. **Hybrid router** after the two-stage stack stops leaving large oracle-vs-natural gaps.
5. **Stop** broad reruns if, after (1), natural Acc@1 gains remain negligible — then invest in **data / geometry** instead.

**Mandatory follow-up command:** `bash outputs/<your_stamp>_shortlist_rerank_upgrade/repro_commands.sh` with higher `--epochs-*` for a serious pass.
