# Phase 2 Experiment Protocol

## Objective
Controlled comparison of three model lines to validate the core hypothesis: explicit structured parsing + soft anchor reasoning helps 3D grounding especially in relation-heavy and anchor-confusion-sensitive cases.

## Experimental Setup

### Data Split
- Primary split: validation set (val)
- If validation set is too small, use a stratified subset of training data for initial runs
- Hold-out test set will be reserved for final evaluation

### Sample Scope
- Initial runs: Limited to 200 samples to establish baseline behavior quickly
- Full runs: As computational resources allow, targeting 1000+ samples for publication-quality results
- All models will be run on the exact same sample set for fair comparison

### Seeds
- Primary seed: 42 (to match previous work and enable reproducibility)
- Additional seeds: 101, 202 (for preliminary robustness assessment if time permits)

### Fixed Hyperparameters
- Learning rate: 1e-4
- Batch size: 8
- Epochs: 5 (sufficient for demonstration, not optimal tuning)
- Weight decay: 0.01
- All other parameters as defined in respective baseline configs

## Outputs to Export

### Primary Metrics
- Acc@1 (primary metric)
- Acc@5 
- Failure rate

### Secondary Metrics  
- Average target margin
- Candidate count statistics
- Per-subset accuracy where applicable

## Model Lines to Compare

1. **Attribute-only baseline** - Object features + target-side language only
2. **Raw-text relation baseline** - Relation-style scoring with raw text embeddings
3. **Structured relation model** - Full pipeline with structured parsing and soft anchor selection

## Evaluation Components

### 1. Overall Comparison
- Acc@1, Acc@5 across all samples
- Performance distribution statistics

### 2. Relation-Stratified Comparison  
- Performance on relation-heavy samples (queries with spatial relation words)
- Performance by specific relation type (left, right, behind, etc.)

### 3. Hard-Case Comparison
- Performance on same-class clutter cases
- Performance on anchor-confusion-sensitive cases
- Performance on occlusion-heavy cases (if applicable)

### 4. Ablation Studies
- Structured model with soft anchor vs. without
- Effect of anchor selection component

### 5. Diagnostic Analysis
- Anchor entropy distribution for structured model
- Top failure categories
- Selected case studies

## Meaningful Improvement Threshold
- Absolute improvement of 0.01+ in Acc@1 is considered meaningful
- Relative improvement of 5%+ is considered meaningful  
- Improvements in hard-case subsets that don't hurt overall performance are especially valuable

## Success Criteria
1. All three models compared on same split with same evaluation
2. Evidence supporting or refuting structured reasoning hypothesis
3. Clear demonstration of where structured approach helps most/least
4. Reproducible and documented experimental procedure