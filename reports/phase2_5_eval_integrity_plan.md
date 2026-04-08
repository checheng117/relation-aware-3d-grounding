# Phase 2.5 Evaluation Integrity Debug Plan

## Overview
The current overall comparison shows identical poor results across all three models (Acc@1 = 0.0000, Acc@5 = 0.1000, Failure Rate = 1.0000), indicating potential data, evaluation, or contract misalignment issues rather than model weakness.

## Potential Causes of Identical Poor Performance

### Data/ID Alignment Issues:
1. Target IDs and candidate IDs use different schemas
2. Ground truth target not actually present in candidate set
3. Mismatched indexing between different components
4. Wrong ID mapping causing all predictions to miss target

### Contract/Batch Issues:
1. Different fields used in prediction records (target_id vs target_object_id)
2. Candidate objects not properly formatted
3. Object mask misalignment
4. Schema mismatches between phases

### Evaluation Issues:
1. Evaluator comparing wrong fields
2. Wrong evaluation contract between model and evaluator
3. ID space mismatches in evaluation logic
4. Filtering causing GT to be excluded

## Verification Plan

### Step 1: Candidate/Target Integrity
- Verify GT target exists in candidate set
- Check ID formats match between target and candidates
- Confirm evaluator compares correct fields

### Step 2: Prediction Contract Audit
- Examine prediction records from each model
- Verify semantic alignment of fields
- Confirm ID spaces are consistent

### Step 3: Human-Readable Case Audit
- Export 20 sample cases with all three model predictions
- Manual inspection of whether predictions are reasonable

### Step 4: Split/Sample Accounting
- Verify actual val split size vs. expected
- Confirm --max-samples behavior
- Ensure all models use identical samples

### Step 5: Bug Fix and Rerun
- Identify and fix any discovered issues
- Rerun minimal comparison to check for improvement

## Success Criteria
A confirmed bug exists if:
- GT target is not in candidate set
- ID mismatches between prediction and evaluation
- Wrong fields being compared
- Different samples used by different models

This plan will isolate whether the poor results stem from infrastructure issues or actual model weakness.