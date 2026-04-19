# Relation-Aware 3D Grounding Benchmark and Methods

A reproducible research codebase for 3D referring-expression grounding on the Nr3D/ReferIt3D benchmark. This project establishes a trustworthy evaluation foundation through dataset recovery, scene-disjoint splitting, and baseline reproduction, then explores relation-aware grounding methods toward the final research target: coverage-calibrated relational reranking.

## Why This Project Matters

Most public 3D grounding repositories suffer from reproducibility problems:

- **Partial datasets**: Released code assumes proprietary or incomplete data
- **Split leakage**: Train/val/test scenes overlap, inflating metrics
- **Unclear protocols**: Metric definitions vary across papers
- **Weak traceability**: No run artifacts, no ablations, no failure analysis

This makes apples-to-apples comparison impossible and published numbers unreliable. This project fixes that by building a fully transparent evaluation pipeline before making method claims.

---

## What Has Been Built

### Dataset Recovery

- Recovered official Nr3D to **41,503 samples** (from incomplete public versions)
- Expanded scene coverage from 269 to **641 ScanNet scenes**
- Validated **zero duplicate samples** across the full dataset
- Preserved original target object IDs and utterance metadata
- Built scene-disjoint splits with verified zero overlap

### Trustworthy Evaluation

- **Scene-disjoint split**: Train/val/test share no scenes
- **Zero overlap verified**: All pairwise scene intersections validated as empty
- **Unified metrics**: Acc@1 and Acc@5 computed consistently across all methods
- **Apples-to-apples comparison**: Same data, same split, same evaluation protocol

### Reproduced Baselines

| Baseline | Test Acc@1 | Test Acc@5 | Status |
|----------|------------|------------|--------|
| ReferIt3DNet | 30.79% | 91.75% | Primary trusted baseline |
| SAT | 28.27% | 87.64% | Secondary baseline (weaker) |

Both baselines reproduced from original papers with scene-disjoint evaluation.

### Custom Methods Explored

| Method | Test Acc@1 | Val Acc@1 | Verdict |
|--------|------------|-----------|---------|
| Parser v1 | 30.04% | — | Discarded (parser noise, unstable fusion) |
| Parser v2 | 28.81% | — | Discarded (below baseline, overfitting) |
| Implicit v1 (dense) | 31.26% | ~33.26% | Promising, crashed (memory overflow) |
| Implicit v2 (sparse top-k) | 28.55% | 31.03% | Discarded (coverage insufficient) |
| Implicit v3 (chunked dense) | 30.36% | 32.90% | Promising but unconfirmed |

---

## Final Benchmark Results

| Method | Test Acc@1 | Test Acc@5 | Status |
|--------|------------:|------------:|--------|
| ReferIt3DNet | 30.79% | 91.75% | Primary baseline |
| SAT | 28.27% | 87.64% | Secondary baseline |
| Implicit v1 | 31.26% | — | Positive signal (incomplete) |
| Implicit v2 | 28.55% | — | Stable but weaker |
| Implicit v3 | 30.36% | — | Promising but unconfirmed |

---

## Key Research Findings

1. **Trustworthy benchmarking matters more than flashy models** — Scene-disjoint splits and reproducible protocols are foundational. Many prior comparisons may be unreliable.

2. **Validation scores alone can mislead** — SAT showed competitive validation performance but underperformed on test. Split leakage masks overfitting.

3. **Parser-based explicit reasoning underperformed** — Heuristic and structured parsers introduce extraction noise. Span grounding is unreliable for spatial reasoning. Hard parser decisions hurt generalization.

4. **Dense pairwise relation modeling showed positive signal** — Implicit v1 achieved +0.47% over baseline before crashing. Val performance suggests dense relations capture useful spatial semantics.

5. **Sparse top-k relation approximation degraded performance** — Implicit v2 with k=5 neighbors lost -2.24% versus baseline. Relation coverage matters; missing long-range evidence hurts.

6. **Chunked dense computation solved memory scaling** — Implicit v3 preserves dense N² semantics with chunked implementation. Numerical equivalence verified. Memory-safe for full scenes.

---

## AAAI Upgrade Path

The current repository is **not yet a paper-ready AAAI main-track method submission**. The trusted contribution is the evaluation and reproduction foundation; the method contribution still needs stable, multi-seed evidence.

### Current Reviewer Verdict

As a strict AAAI reviewer, the current method state would be:

| Submission Framing | Likely Verdict | Reason |
|--------------------|----------------|--------|
| Current method paper | Reject / Weak Reject | Custom methods do not yet provide stable, confirmed gains over ReferIt3DNet |
| v3 only, if it reaches ~31.3–31.8% | Weak Reject / Borderline | Small engineering gain, limited novelty unless supported by stronger diagnostics |
| COVER-3D with 3-seed overall + hard-subset gains | Accept | Coverage and calibration claims become testable and supported |
| COVER-3D with cross-backbone gains and full diagnostics | Strong Accept potential | Method, protocol, and diagnostic contributions reinforce each other |

### Target Paper Direction

Working title:

**COVER-3D: Coverage-Calibrated Relational Grounding for Scene-Disjoint 3D Referring Expressions**

Core claim:

Hard relational failures in 3D grounding are caused by two coupled problems:

1. **Coverage failure**: sparse/top-k relation neighborhoods miss useful anchors, especially in long-range, multi-anchor, and same-class clutter cases.
2. **Calibration failure**: relation branches can overpower reliable base predictions when parser, anchor, or relation evidence is noisy.

The final method should show that **dense relation coverage plus uncertainty-calibrated fusion** improves hard relational subsets while preserving overall accuracy.

### Minimum Acceptance Bar

The project should only be positioned as a AAAI method paper if all of the following hold:

- Overall Acc@1 improves over the trusted ReferIt3DNet baseline by at least **+1.2 points** (target: **32.0%+** vs 30.79%).
- Hard relational, same-class clutter, and long-range anchor subsets improve by roughly **+3 to +5 Acc@1 points**.
- Acc@5 remains close to the baseline level and does not collapse.
- Results hold across **at least 3 seeds** with mean/std and statistical tests where differences are small.
- Gains are not restricted to a weak baseline or a single lucky run.
- Ablations show that both **coverage** and **calibration** are necessary.
- Qualitative cases match the quantitative story: dense coverage finds useful anchors; calibration prevents noisy relation evidence from hurting easy cases.

### COVER-3D Roadmap

1. **Freeze the official scene-disjoint protocol** using `data/processed/scene_disjoint/official_scene_disjoint`.
2. **Implement coverage diagnostics** before more training: coverage@k, anchor reachability, long-range anchor rank, same-class clutter, shared-anchor negatives.
3. **Build a model-agnostic backbone adapter** that exposes `base_logits`, `object_embeddings`, `object_geometry`, `candidate_mask`, and `utterance_features`.
4. **Convert relation modeling into a reranker** rather than another standalone backbone.
5. **Reuse chunked dense all-pair relation computation** as the coverage-preserving primitive.
6. **Add calibrated fusion** using base margin, anchor entropy, parser confidence, relation margin, and relation evidence strength.
7. **Add hard relational training hooks** for same-class distractors, shared anchors, long-range anchors, relation counterfactuals, paraphrases, and ambiguous low-margin cases.
8. **Run 3-seed formal experiments on stable hardware only**.
9. **Report main, hard-subset, ablation, paraphrase, runtime/memory, and failure-taxonomy results**.
10. If the acceptance bar is not met, reposition the project as a reproducibility and diagnostic evaluation paper rather than a method paper.

---

## Repository Structure

```
relation-aware-3d-grounding/
├── src/rag3d/               # Main Python package
│   ├── datasets/            # Nr3D/ScanNet loading, schemas, builders
│   ├── encoders/            # Object, point, and fusion encoders
│   ├── models/              # ReferIt3DNet, SAT, relation-aware variants
│   ├── parsers/             # Heuristic, structured, VLM parser adapters
│   ├── relation_reasoner/   # Relation scoring, reranking, anchors
│   ├── evaluation/          # Metrics, stratified eval, paraphrase eval
│   ├── diagnostics/         # Failure taxonomy, confidence, case analysis
│   └ visualization/         # Scene visualization, qualitative panels
│   └── training/            # Training runner, checkpoint handling
├── configs/                 # Dataset, training, evaluation configs
├── scripts/                 # Data prep, training, evaluation entrypoints
├── repro/                   # Baseline reproduction code
│   ├── referit3d_baseline/  # ReferIt3DNet reproduction
│   └ sat_baseline/          # SAT reproduction
├── reports/                 # Experiment reports, audits, analyses
├── data/                    # Local data (git-ignored)
└── tests/                   # Unit and smoke tests
```

---

## How To Reproduce

### Setup

```bash
conda env create -f environment.yml
conda activate rag3d
pip install -e ".[dev,viz]"

make check-env
make test
```

### Data Preparation

```bash
# Fetch Nr3D annotations from HuggingFace
python scripts/fetch_nr3d_hf.py

# Fetch ScanNet aggregation assets
python scripts/fetch_scannet_aggregations.py --artifact aggregation-json

# Build scene-disjoint splits
python scripts/prepare_data.py \
  --mode build-nr3d-official-scene-disjoint \
  --config configs/dataset/referit3d_scene_disjoint.yaml

# Validate split integrity
python scripts/validate_scene_disjoint_splits.py \
  --manifest-dir data/processed/scene_disjoint/official_scene_disjoint

# Prepare text features
python scripts/prepare_bert_features.py \
  --manifest-dir data/processed/scene_disjoint/official_scene_disjoint

# Prepare object features
python scripts/compute_object_features.py \
  --config configs/dataset/expanded_nr3d.yaml
```

### Baseline Reproduction

ReferIt3DNet:

```bash
python repro/referit3d_baseline/scripts/train.py \
  --config repro/referit3d_baseline/configs/learned_class_embedding.yaml \
  --device cuda

python repro/referit3d_baseline/scripts/evaluate.py \
  --config repro/referit3d_baseline/configs/learned_class_embedding.yaml \
  --device cuda
```

SAT:

```bash
python repro/sat_baseline/scripts/train_sat.py \
  --config configs/sat_baseline.yaml \
  --device cuda
```

### Relation-Aware Methods

```bash
# Smoke test
python scripts/smoke_test_implicit_v3.py

# Training
python scripts/train_implicit_relation_v3.py \
  --config configs/implicit_relation_v3.yaml \
  --device cuda

# Evaluation
python scripts/evaluate_implicit_relation_v3.py \
  --config configs/implicit_relation_v3.yaml \
  --device cuda
```

---

## Reports and Artifacts

Detailed experiment reports are stored in [reports/](reports/):

| Report | Description |
|--------|-------------|
| [final_method_status.md](reports/final_method_status.md) | Method comparison and verdicts |
| [scene_disjoint_split_recovery_results.md](reports/scene_disjoint_split_recovery_results.md) | Split repair and integrity validation |
| [implicit_relation_v3_archive.md](reports/implicit_relation_v3_archive.md) | v3 findings, crash context, continuation path |
| [next_phase_research_plan.md](reports/next_phase_research_plan.md) | Phase transition and research roadmap |

---

## Current Recommended Usage

- **Use ReferIt3DNet baseline** as the official comparison anchor (30.79% test Acc@1)
- **Use Implicit v3** only as evidence that chunked dense relation computation is feasible; do not claim superiority
- **Use COVER-3D** as the planned final method direction: model-agnostic dense relational reranking with calibrated fusion
- **Do not use Parser v1/v2 or Implicit v2** — these methods are discarded with clear evidence

---

## Hardware Limitations Encountered

The development machine exhibited GPU driver instability during v3 training:

- Multiple system crashes with NVIDIA driver hangs
- Short diagnostic runs completed successfully
- Resume runs crashed regardless of mitigation attempts
- This is a hardware issue, not a code or model problem

**Implication**: Final v3 validation requires stable hardware. The current result is promising but unconfirmed.

---

## Future Roadmap

### Immediate Priority

1. Align README, `.claude`, and reports around the AAAI upgrade path
2. Implement relation coverage diagnostics and hard-case tagging
3. Define the model-agnostic backbone adapter contract
4. Implement calibrated COVER-3D reranking as the final method path

### Medium Priority

5. Run CPU/unit/smoke validation locally
6. Move formal 3-seed training to stable GPU hardware
7. Complete ablations: sparse vs dense, calibrated vs uncalibrated, hard negatives, paraphrase consistency, oracle anchor, noisy parser stress

### Long-Term

8. Cross-backbone validation beyond ReferIt3DNet
9. Multi-view geometry integration
10. Real 3D spatial reasoning beyond bounding-box features

---

## Development Principles

- Preserve the trusted evaluation base before making method claims
- Do not report unconfirmed results as final improvements
- Prefer stratified and diagnostic evaluation over single-number accuracy
- Treat parser outputs as noisy weak signals, not oracle structure
- Keep generated data, checkpoints, and feature caches out of git
- Any paper claim requires run artifacts, ablations, and failure analysis

---

## License

MIT License. See [LICENSE](LICENSE).
