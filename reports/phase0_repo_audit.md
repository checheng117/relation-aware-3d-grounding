# Phase 0 Repo Audit Report

## Repo Current Modules Overview

Current directory: /home/cc/Project/CC/relation-aware-3d-grounding

### Key Source Modules
- `src/rag3d/` - Main codebase
  - `datasets/` - Data handling (referit3d.py, scannet_objects.py, builder.py, transforms.py, collate.py)
  - `evaluation/` - Evaluation logic (metrics.py, evaluator.py, stratified_eval.py, two_stage_eval.py, etc.)
  - `relation_reasoner/` - Core models (attribute_scorer.py, relation_scorer.py, text_encoding.py, etc.)
  - `training/` - Training infrastructure (runner.py)
  - `utils/` - Utilities (config.py, io.py, logging.py, seed.py, env.py)
  - `parsers/` - Various parsers (cached_parser.py, base.py, heuristic_parser.py, etc.)

### Scripts Available
- `scripts/` directory with various training/evaluation scripts:
  - `train_baseline.py` - For attribute-only and raw-text relation models
  - `train_main.py` - Main training script
  - `eval_all.py` - Evaluation script
  - `check_env.py`, `smoke_test.py`, `setup_env.sh` - Setup and testing scripts

### Existing Shortlist->Rerank Components
- Shortlist creation (coarse stage 1)
- Re-ranking logic
- Two-stage training/evaluation pipelines
- Various checkpoints and metrics already in outputs/

### Models Available
- AttributeOnlyModel (attribute-only baseline)
- RawTextRelationModel (raw-text relation baseline)
- Relation-aware structured scorers

## Reusable Modules

### Can Be Directly Reused
- `src/rag3d/datasets/referit3d.py` - Dataset class
- `src/rag3d/datasets/scannet_objects.py` - Object representations
- `src/rag3d/datasets/collate.py` - Collate functions
- `src/rag3d/training/runner.py` - Training infrastructure
- `src/rag3d/utils/config.py` - Config handling
- Some evaluation metrics from `src/rag3d/evaluation/`

### Need Wrappers/Adapters
- Current data schema needs unification
- Evaluation system needs standardization
- Training interfaces need harmonization

## New Modules to Add

### For Phase 0 Foundation
- `src/rag3d/datasets/schema.py` - Unified data schema
- `src/rag3d/datasets/adapters.py` - Adapters for schema transformation
- `src/rag3d/evaluation/metrics.py` - Standardized metrics
- `src/rag3d/evaluation/stratified_eval.py` - Stratified evaluation
- `src/rag3d/diagnostics/failure_taxonomy.py` - Failure analysis
- `src/rag3d/diagnostics/tagging.py` - Heuristic tagging
- `scripts/eval_foundation.py` - Foundation evaluation script
- `scripts/run_attribute_baseline.py` - Clean baseline script

## Planned Minimal Intrusive Implementation

### Strategy
1. Add new modules without changing existing code significantly
2. Create adapters to bridge old and new schemas
3. Introduce a new, clean evaluation pathway that works alongside existing ones
4. Provide clear interfaces for future structured parser/anchor/relation components

### Interface Design
- New schema classes will be lightweight dataclasses
- Adapter functions will convert between old and new formats
- New evaluation will work with both old and new model interfaces
- Gradual migration path without breaking changes

### Integration Approach
- Add schema definition files without touching existing data loaders
- Add new evaluation functions that can work with current model outputs
- Provide backward compatibility through adapters
- Maintain current training flows while introducing cleaner alternatives