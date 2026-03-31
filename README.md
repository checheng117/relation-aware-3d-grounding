# Relation-Aware 3D Grounding (Summary README)

This repository is a research codebase for structured 3D grounding on ReferIt3D-style data.  
Current focus is the two-stage `shortlist -> rerank` stack and its end-to-end robustness under corrected evaluation.

---

## Project Snapshot

- **Task**: predict target object from scene objects + language.
- **Model lines**: attribute baseline, raw-text relation, relation-aware structured scorer.
- **Main engineering track**: shortlist quality + reranker alignment under full-scene evaluation.
- **Output style**: checkpointed runs with report-ready CSV/JSON/MD/PNG artifacts under `outputs/<timestamp>_*`.

---

## Quick Start

### Environment

```bash
git clone <your-repo-url> && cd relation-aware-3d-grounding
make env
conda activate rag3d
make test && make smoke
```

### Data

```bash
python scripts/prepare_data.py --mode validate --config configs/dataset/referit3d.yaml
python scripts/prepare_data.py --mode build --config configs/dataset/referit3d.yaml
```

For development without real scans:

```bash
python scripts/prepare_data.py --mode mock-debug
```

### Core Training/Eval Entrypoints

- Single-line training: `scripts/train_baseline.py`, `scripts/train_main.py`
- Two-stage training: `scripts/train_coarse_stage1.py`, `scripts/train_two_stage_rerank.py`
- Main eval: `scripts/eval_all.py`
- Two-stage metric helpers: `src/rag3d/evaluation/two_stage_eval.py`, `src/rag3d/evaluation/two_stage_rerank_metrics.py`

---

## Validated Results Timeline (Current Reference)

| Phase | Key validated outcome |
|---|---|
| Fix combined eval + natural-shortlist loss | corrected reranker row improves end-to-end (`0.0256 -> 0.0385`) |
| Official shortlist strengthening | Recall@20 `0.5513 -> 0.7821`; corrected two-stage Acc@1 improves with improved shortlist |
| Reranker rebalance on improved shortlist | `improved_shortlist_plus_reference_rerank = 0.1090`; retrained reranker matches but does not exceed |
| Minimal co-adaptation | intermediate row reaches `0.1154`; naive strict second pass falls back to `0.1090` |
| Conservative second-pass | `low_lr_secondpass = 0.1154` (retains gain), other conservative variants return around `0.1090` |
| Gain-retention validation | extra seed for low-LR drops to `0.1026`; local LR sweep around `5e-6` stays at `0.1154` for validated seed |

Authoritative recent bundles:

- `outputs/20260331_183805_conservative_secondpass/`
- `outputs/20260331_185900_gain_retention_validation/`

---

## Current Conclusion

- **Established**: shortlist strengthening, reranker rebalance, co-adaptation signal, and seed-42 low-LR retention are all real.
- **Not established**: broad seed-robust retained gain (extra seed in narrow validation dropped to `0.1026`).
- **Best retained reference row** (validated seed): `0.1154`.
- **Practical baseline reference**: `improved_shortlist_plus_reference_rerank = 0.1090`.

In short: there is a credible retained-gain regime, but seed robustness remains limited.

---

## Recommended Next Action

Given current evidence and scope control:

- **Stop broad experimentation and move to writing/reporting**.
- Frame retained gain as **locally stable (around low LR on validated seed)** but **not yet fully seed-stable**.

---

## Repository Layout

| Path | Role |
|---|---|
| `src/rag3d/` | models, losses, evaluation, diagnostics |
| `scripts/` | training/eval orchestration and phase runners |
| `configs/` | dataset/train/eval YAMLs |
| `reports/` | phase plans, protocol notes, summaries |
| `outputs/` | timestamped experiment artifacts |

---

## License

MIT — see `LICENSE`.
