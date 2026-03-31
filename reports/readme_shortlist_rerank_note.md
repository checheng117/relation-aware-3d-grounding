# README note — shortlist / rerank phase

## What changed

- Under **Staged roadmap**, row **C** now describes the **shortlist-aware retrieval + rerank upgrade** script phase instead of a vague “align coarse training” bullet.
- Added a footnote that **phase C is the primary bottleneck attack** and the **hybrid router is secondary**.
- Added a short subsection **Shortlist + rerank upgrade (current engineering focus)** pointing to timestamped `outputs/*_shortlist_rerank_upgrade/` and `reports/shortlist_rerank_upgrade_summary.md`.

## Why

The hybrid-router experiments showed **limited oracle branch ceiling** and confirmed that **shortlist recall + rerank** dominate; the README now states that the project is **actively prioritizing** that stack rather than implying equal weight to hybrid routing.
