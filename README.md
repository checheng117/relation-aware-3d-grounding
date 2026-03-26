# Relation-Aware 3D Grounding for Embodied Perception

Structured 3D perception bridge: given a **set of objects** in a scene (with optional geometric cues) and a **natural-language utterance**, the system predicts a **target object**, a **soft anchor distribution**, relation-oriented signals, confidence, and **diagnostic tags**. The design targets **embodied perception stacks** that need an explicit, inspectable grounding module—not GUI grounding, post-training of code LLMs, or long-horizon planning.

---

## Core idea

End-to-end flow:

1. **Object Set Builder** — manifests from ScanNet-style aggregations + utterance tables (BYO data).
2. **Object Encoder** — compact MLP features (+ optional geometry / quality context).
3. **Structured Language Parser** — heuristic pipeline with optional hooks; exposes structured cues instead of pooling the whole utterance.
4. **Soft Anchor Selector** — distribution over candidate objects for relational language.
5. **Relation-Robust Target Scorer** — compares **attribute-only**, **raw-text relation**, and **relation-aware** hypotheses.
6. **Diagnostics & Visualization** — margins, entropy, failure taxonomies, case exports, report-oriented tables.

Three **model lines** are implemented for controlled comparison: attribute-only, raw-text relation pooling, and relation-aware structured grounding.

---

## Why this matters

Standard 3D grounding metrics alone do not answer what an embodied system needs: **which object is the anchor**, **which relation fired**, and **why** the model failed. This repository treats grounding as a **diagnosable bridge** with a stable JSON-oriented output contract (`BridgeModuleOutput` in `src/rag3d/datasets/schemas.py`, `src/rag3d/diagnostics/bridge_output.py`). A central empirical theme is **candidate-space scaling**: when every scene object is a candidate, weak geometry and label noise dominate; relation-aware structure helps most clearly in **controlled** candidate regimes and in analysis slices—not as a magic fix for unconstrained full-scene collapse.

---

## Features

- Unified **object / parser / bridge-output** schemas and manifest-driven data loading.
- **Relation-aware** scoring with soft anchors vs. attribute-only and raw-text baselines.
- Optional **hard-negative** (same-class) hinge in training (`loss.hard_negative` in train YAML; see `configs/train/blueprint_loss_example.yaml`).
- **Paraphrase consistency** evaluation: `python scripts/eval_paraphrase_consistency.py --help`.
- **Stratified metrics** — relation-type slices, clutter / occlusion / anchor-confusion heuristics, parser-failure and **low logit-margin** subsets (`stratified_results.json`).
- **Two-stage shortlist → rerank** pipeline — coarse top-K plus relation head on the shortlist; stage-1 recall and shortlist quality are first-class evaluation targets (`scripts/train_coarse_stage1.py`, `scripts/train_two_stage_rerank.py`, `scripts/eval_stage1_recall_pass.py`, `src/rag3d/evaluation/shortlist_promote.py`).
- **Report-ready outputs** — `make figures`, `python scripts/collect_results.py`, optional `python scripts/build_report_bundle_blueprint.py` (copies key metrics CSV/JSON under `outputs/report_bundle_blueprint/` and `outputs/figures/report_ready_blueprint/`).

---

## Repository layout

| Path | Role |
|------|------|
| `src/rag3d/` | Models, data loaders, losses, evaluation, diagnostics |
| `configs/` | Dataset, train, and eval YAML |
| `scripts/` | Data prep, training, eval, figures, aggregation utilities |
| `tests/` | PyTest suite |
| `data/raw/` | Your ReferIt3D / ScanNet-style trees (not shipped) |
| `data/processed/` | Generated manifests (`prepare_data.py`) |
| `outputs/` | Checkpoints, metrics JSON, figures (typically gitignored) |
| `docs/DATASET_SETUP.md` | Data layout and `prepare_data.py` commands |
| `docs/CSC6133_Upgraded_3D_Spatial_Reasoning_Blueprint_CheCheng.docx` | Optional archived specification (see `scripts/extract_blueprint_docx.py`) |

---

## Installation (Conda, recommended)

**Python 3.10**, environment name **`rag3d`** (see `environment.yml`).

```bash
git clone <your-repo-url> && cd relation-aware-3d-grounding
make env          # runs scripts/setup_env.sh: conda env + pip install -e ".[dev,viz]"
conda activate rag3d
make test && make smoke
```

Optional: `python scripts/check_env.py`. If `import torch` fails on broken CUDA drivers, the Makefile already sets `CUDA_VISIBLE_DEVICES=` for `test` / `smoke` / `figures`; you can run `CUDA_VISIBLE_DEVICES= python scripts/check_env.py` similarly.

**Secrets:** copy `.env.example` to `.env` and set `HF_TOKEN` only if you pull gated Hugging Face assets. Never commit `.env`.

For **GPU-specific PyTorch** builds, install the official CUDA wheel into the activated `rag3d` environment and re-run `check_env.py`. `requirements.txt` is a pip-only reference; Conda remains the supported path.

---

## Data setup

This project does **not** redistribute ReferIt3D or ScanNet. Place data under `data/raw/referit3d/` (or point `configs/dataset/referit3d.yaml` at your root). See **[docs/DATASET_SETUP.md](docs/DATASET_SETUP.md)** for directory layout, CSV columns, and commands.

```bash
python scripts/prepare_data.py --mode validate --config configs/dataset/referit3d.yaml
python scripts/prepare_data.py --mode build --config configs/dataset/referit3d.yaml
```

**Without real scans**, use mock manifests for development:

```bash
python scripts/prepare_data.py --mode mock-debug
```

---

## Training and evaluation (quickstart)

**Debug / mock path:**

```bash
python scripts/prepare_data.py --mode mock-debug
python scripts/train_baseline.py --config configs/train/debug_baseline.yaml
python scripts/train_baseline.py --config configs/train/debug_raw_relation.yaml
python scripts/train_main.py --config configs/train/debug_main.yaml
python scripts/eval_all.py --config configs/eval/debug.yaml
python scripts/analyze_hard_cases.py --use-debug-subdir
make figures
python scripts/collect_results.py
```

**Real data:** after manifests exist, train with `configs/train/baseline.yaml`, `raw_relation.yaml`, and `main.yaml` (`mode: real` as appropriate), then `python scripts/eval_all.py --config configs/eval/default.yaml`, `make figures`, and `collect_results.py`.

**Useful artifacts:** checkpoints under `outputs/checkpoints/`, metrics under `outputs/metrics/` (`main_results.json`, `stratified_results.json`), curated tables under `outputs/figures/report_ready/`, hard-case JSON under `outputs/case_studies/hard_case_summary.json` (includes `BridgeModuleOutput` when using `analyze_hard_cases.py`).

---

## Main findings (honest summary)

Empirical work in this codebase supports the following **qualitative** conclusions (exact numbers depend on your data split and configs):

- **Relation-aware modeling** can outperform raw-text relation and attribute-only baselines when the **candidate set is controlled** (e.g., entity-aligned lists); the effect is much harder to see under **full-scene** candidate explosion.
- **Full-scene** accuracy is often dominated by **weak or incomplete geometry** (synthetic features when OBBs are missing) and **many-way classification**, not by a single missing training trick.
- **Two-stage** coarse shortlist + rerank is only as strong as **stage-1 recall and shortlist alignment**; promoting coarse checkpoints by a single long-tail recall metric can **mismatch** K-aligned rerank utility—see `shortlist_aligned_score` in `src/rag3d/evaluation/shortlist_promote.py`.

Illustrative **full-scene** numbers from one bundled CSV (`outputs/figures/main_results_table.csv`, n=156) are on the order of a few percent Acc@1 across lines—use them as a sanity reference, not as a benchmark claim.

---

## Limitations

- Encoders are **lightweight** (MLP + hashed text); this is a **methodology and diagnostics** codebase, not a state-of-the-art ReferIt3D listener leaderboard entry.
- **Geometry** is incomplete for many objects (missing OBBs); features fall back to **placeholders**, which limits metric shape under full-scene evaluation.
- **Full-scene grounding** remains **hard** in this stack; reported gains are **slice- and regime-dependent**.
- **Shortlist / stage-1 quality** is the main bottleneck for two-stage pipelines; rerank cannot recover targets absent from the coarse top-K.

---

## Future work

- Richer **object geometry** (complete OBB coverage or point-cloud encoders) with the same diagnostic harness.
- **Calibration** and abstention on bridge outputs for planner-facing APIs.
- Stronger **language parsing** (e.g., optional VLM) while keeping the structured bridge contract.
- Tighter **integration tests** with downstream modules that consume `BridgeModuleOutput` without re-implementing grounding.

---

## License

MIT — see [LICENSE](LICENSE).
