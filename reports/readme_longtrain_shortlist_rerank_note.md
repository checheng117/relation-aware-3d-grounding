# README change — long-train shortlist + rerank (for maintainers)

A short subsection was added to the root **README** under the roadmap / phase narrative:

- The project is now also tracking a **long-train shortlist + rerank strengthening** phase.
- **Checkpoint selection** for coarse (when enabled) and rerank training is tied to **natural two-stage validation Acc@1**, not coarse Recall@K alone.
- The **hybrid B/C router** is described as **secondary** until the **core shortlist / rerank bottleneck** shows clearer gains under this protocol.

Artifacts: `outputs/<timestamp>_longtrain_shortlist_rerank/` plus `reports/long_train_shortlist_rerank_plan.md` and `reports/longtrain_shortlist_rerank_summary.md`.
