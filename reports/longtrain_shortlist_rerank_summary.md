# Long-train shortlist + rerank phase — summary

## 1. What was implemented

- **Checkpoint selection**
  - Coarse: optional YAML `val_two_stage_selection` → logs **`val_pipeline_natural_acc@1`** and saves **`{run_name}_best_pipeline_natural.pt`** (`src/rag3d/training/runner.py`, `src/rag3d/training/checkpoint_selection.py`, `scripts/train_coarse_stage1.py`).
  - Rerank: per-epoch **natural + oracle** shortlist metrics via `eval_two_stage_inject_mode`; saves **`{run_name}_best_natural_two_stage.pt`** (`scripts/train_two_stage_rerank.py`).
- **`TwoStageCoarseRerankModel`**: optional **`freeze_coarse`** so selection code can wrap the **live** coarse module without disabling its gradients for the outer trainer.
- **Orchestrator** `scripts/run_longtrain_shortlist_rerank_phase.py`: timestamped `outputs/<stamp>_longtrain_shortlist_rerank/` with configs, logs, tables, figures, `repro_commands.sh`, and `report_bundle/README.md`.
- **Templates:** `configs/train/rerank/rerank_longtrain_{natural,oracle}.yaml`, `configs/train/coarse/coarse_focused_hardneg_longtrain.yaml`.
- **Documentation:** `reports/long_train_shortlist_rerank_plan.md`, `reports/checkpoint_selection_note.md`, `reports/focused_shortlist_hard_negative_note.md`, `reports/readme_longtrain_shortlist_rerank_note.md`, README subsection.

## 2. Prior assets reused

- Two-stage eval helpers (`two_stage_eval.py`, `two_stage_rerank_metrics.py`, `coarse_recall.py`).
- Training entrypoints `train_coarse_stage1.py` / `train_two_stage_rerank.py`.
- Upgrade-phase patterns from `run_shortlist_rerank_upgrade_phase.py` (tables, combined JSON, matplotlib **Agg**).

## 3. How checkpoint selection changed

- **Primary** metric for **best** artifacts is **natural two-stage full-scene val Acc@1** (coarse selection uses a **frozen reference reranker**; rerank selection uses the **trained** two-stage model’s own fine head).
- Coarse **Recall@K** remains **logged** for diagnosis but is **not** the sole promotion criterion in this phase.

## 4. Did longer reranker training materially help?

- A **full official run** should use the defaults in `run_longtrain_shortlist_rerank_phase.py` (**12** rerank epochs, **8** coarse epochs) or your hardware budget.
- A **smoke run** on this machine (`--epochs-rerank 1 --epochs-coarse 1`, stamp **`20260327_230719`**) showed **natural** val Acc@1 moving from **~0.032** (baseline pipeline A) to **~0.045** (row **B** / **D**) when swapping in **`rerank_longtrain_natural_best_natural_two_stage.pt`** — a modest **+0.013** absolute on **n=156** val; **not** conclusive for “long train” until the longer schedule is executed.
- **Oracle** shortlist Acc@1 stayed **well above** natural (**~0.37–0.39** vs **~0.03–0.045** in that smoke bundle), so **shortlist + rerank** remain a **mixed** bottleneck.

## 5. Did focused hard negatives materially help retrieval?

- In the same **1-epoch** coarse smoke, **row C** (focused coarse + old rerank) **matched** baseline **A** on natural Acc@1 (**0.032**); Recall@K movement should be read from `shortlist_retrieval_results_longtrain.json` in that output tree. **One epoch is insufficient** to claim stable retrieval gains.

## 6. Did the combined pipeline improve natural full-scene Acc@1?

- Smoke: **B** and **D** **did** improve vs **A**; **C** did **not** on this short budget.
- Re-run with **12/8** epochs and treat **`shortlist_rerank_combined_results_longtrain.json`** in the new timestamp folder as the **official** readout.

## 7. Next best step after this phase

| Situation | Suggested direction |
|-----------|---------------------|
| Natural Acc@1 **stays** far below oracle after long train | Continue **shortlist work** (Recall@K, K choice) **and** **stronger rerank** (capacity / training signal). |
| Oracle Acc@1 **stays** moderate while conditional-in-K is low | **Rerank** is still weak — invest in **reranker** before hybrid. |
| Natural tracks oracle gains | Retrieval was binding; **curriculum** on coarse may help. |
| Gains remain **tiny** across long budgets | Pause broad refactors; consider **geometry / features** (your **phase D**, explicitly **not** touched here) or **stop** if stack is data-limited. |
| Core pipeline healthy | **Reintroduce hybrid router** on top of the stronger base. |

**Protocol M (mixed oracle/natural shortlist per batch)** was **not** implemented (would need a per-sample injection probability in forward); documented in `oracle_reranker_results_longtrain.json`.
