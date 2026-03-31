# Checkpoint selection — previous vs new behavior

## Previous behavior

### Coarse (stage-1)

- `scripts/train_coarse_stage1.py` uses `run_training_loop`, which logs **full-scene coarse** `val_acc@1`, optional **Recall@K** (`val_coarse_recall@*`), and slice keys when `val_coarse_recall_ks` is set.
- **Every epoch** is saved as `{run_name}_epoch{e}.pt`; `{run_name}_last.pt` is the final epoch.
- Downstream practice in some sweeps (e.g. `scripts/promote_coarse_opt_rerank.py`) **ranks coarse candidates by retrieval metrics** such as `recall@20` / `recall@10` to pick which coarse checkpoint gets a rerank head. That optimizes **stage-1 recall**, not necessarily **natural two-stage Acc@1** with a fixed reranker.

### Reranker (two-stage fine)

- `scripts/train_two_stage_rerank.py` previously computed a **batch argmax** `val_acc@1` on full-scene scattered logits. That is **already** aligned with the natural pipeline when the model is in **eval** mode (no gold injection into the shortlist), but the metric was **not** named as an end-to-end objective and **no explicit `best_*` checkpoint** was written when it improved.
- Operators often defaulted to `*_last.pt`, which can be **suboptimal** if the final epoch overfits or drifts on the true objective.

## New behavior

### Coarse

- YAML block **`val_two_stage_selection`** (parsed in `train_coarse_stage1.py`) enables, each epoch, a pass of **`evaluate_coarse_with_fixed_rerank`** (`src/rag3d/training/checkpoint_selection.py`):
  - Builds a **two-stage** model wrapping the **current training coarse weights** and a **frozen fine** loaded from `reference_rerank_checkpoint` (fine.* tensors from a two-stage `.pt`).
  - Evaluates **natural** shortlist (`inject_gold_in_shortlist=False`) and logs **`val_pipeline_natural_acc@1`** as the **primary selection metric**.
  - Also logs oracle-shortlist pipeline Acc@1 / MRR and conditional-in-K style diagnostics for interpretation.
- Whenever **`val_pipeline_natural_acc@1`** improves, the runner saves **`{run_name}_best_pipeline_natural.pt`** (full coarse `state_dict`).

### Reranker

- Each epoch, **`eval_two_stage_inject_mode`** runs on the validation loader:
  - **Natural** and **oracle** shortlist metrics, **MRR**, **shortlist recall**, **conditional Acc given gold ∈ K**.
- **`val_acc@1`** in the JSONL is set equal to **`val_natural_two_stage_acc@1`** so the headline column matches the selection objective.
- When **natural** two-stage val Acc@1 improves, the script saves **`{run_name}_best_natural_two_stage.pt`**.

## Why this is better aligned with the goal

The deployable system is **coarse top-K → rerank on shortlist → full-scene argmax**. Promoting models using **only** coarse Recall@K can improve shortlist coverage while **hurting** the logits that reranking actually sees, or can plateau **natural** accuracy if the reranker is the binding constraint. Driving selection from **natural two-stage val Acc@1** (and saving explicit **best** checkpoints) matches **the metric users care about** and stabilizes comparison across retrieval and rerank experiments.

## Legacy scripts

- **`scripts/promote_coarse_opt_rerank.py`** still sorts by **recall@20** for historical optimization sweeps. For this phase, treat **`_best_pipeline_natural.pt`** (coarse) and **`_best_natural_two_stage.pt`** (rerank) as authoritative when those training modes were enabled.
