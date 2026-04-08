# Phase 1 Relation Stack Audit Report

## Overview
Audit of current foundation implementation and available modules for building the relation stack pipeline.

## Inspected Foundation Modules

### 1. Data Schema (`src/rag3d/datasets/schema.py`)
- ✅ Clean, well-defined schema classes
- ✅ GroundingSample with all required fields
- ✅ ObjectRecord with proper structure
- ✅ Good for reuse in new models

### 2. Adapters (`src/rag3d/datasets/adapters.py`)
- ✅ Well-implemented adapter functions
- ✅ Compatible with new structured approach
- ✅ Good base for extending with new data formats

### 3. Evaluation (`src/rag3d/evaluation/metrics.py`)
- ✅ Comprehensive metric functions
- ✅ Multiple export formats (JSON, CSV, Markdown)
- ✅ Good base for new model outputs
- ✅ Maintains backward compatibility

### 4. Stratified Evaluation (`src/rag3d/evaluation/stratified_eval.py`)
- ✅ Handles relation-heavy and clutter cases
- ✅ Tagging functions available
- ✅ Good framework for comparing model types

### 5. Diagnostics (`src/rag3d/diagnostics/failure_taxonomy.py`)
- ✅ Clear failure classification system
- ✅ Good for analyzing model differences
- ✅ Ready for anchor-related failures

### 6. Diagnostics (`src/rag3d/diagnostics/tagging.py`)
- ✅ Heuristic tagging functions
- ✅ Handles same-class clutter and relation-heavy cases
- ✅ Ready for new model types

## Existing Model Code Inspection

### Checked `src/rag3d/relation_reasoner/` directory:
Found existing modules:
- `attribute_scorer.py` - AttributeOnlyModel implementation
- `relation_scorer.py` - Contains RawTextRelationModel 
- `text_encoding.py` - Text encoding utilities
- `anchor_selector.py` - Already exists! Basic anchor selection
- `geom_context.py` - Geometry context utilities

### Key Finding: `relation_scorer.py` already contains RawTextRelationModel
This model can be reused directly for Step 1, potentially requiring minimal adaptation.

### Key Finding: `anchor_selector.py` already exists
This may already implement soft anchor selection - need to examine it.

## Reusable Modules Identified

### Direct Reuse:
1. `src/rag3d/relation_reasoner/attribute_scorer.py` - For attribute-only baseline
2. `src/rag3d/relation_reasoner/relation_scorer.py` - RawTextRelationModel exists
3. `src/rag3d/relation_reasoner/text_encoding.py` - Text encoding utilities
4. `src/rag3d/relation_reasoner/anchor_selector.py` - May already have soft anchor selection
5. `src/rag3d/encoders/` - Object encoding modules
6. All evaluation and diagnostic modules

### Adaptable:
1. Training infrastructure from existing scripts
2. Data loading and collation from existing modules

## What New Modules Are Needed

### 1. Updated Raw-Text Relation Component
- Potentially just adapting existing RawTextRelationModel to new evaluation pipeline
- New training/inference script for consistency with foundation

### 2. Structured Parser Components
- `src/rag3d/parsers/structured_parser.py` - Structured parsing logic
- `src/rag3d/parsers/parser_cache.py` - Cache mechanism
- New heuristic parser if needed

### 3. Enhanced Anchor Selection (if needed)
- If existing anchor_selector.py is sufficient, may only need integration
- Otherwise, implement new soft anchor selector

### 4. Structured Relation Model
- `src/rag3d/relation_reasoner/structured_relation_model.py` - Full pipeline

## Minimal-Intrusion Implementation Plan

### Phase 1A: Raw-Text Relation Baseline
1. Adapt existing RawTextRelationModel from relation_scorer.py to work with new schema
2. Create new training/inference script following attribute baseline pattern
3. Verify it works with unified evaluation

### Phase 1B: Parser Cache Layer
1. Implement parser cache system
2. Create structured parser interface
3. Add heuristic fallback parser

### Phase 1C: Soft Anchor Selector Integration
1. Examine existing anchor_selector.py
2. Extend if needed for structured outputs
3. Integrate with evaluation system

### Phase 1D: Structured Relation Model
1. Combine parser output, anchor selector, and relation scoring
2. Ensure compatible outputs with other models
3. Connect to evaluation pipeline

### Phase 1E: Comprehensive Comparison
1. Run all three models through evaluation
2. Generate comparison artifacts
3. Document findings

This approach maximizes reuse of existing code while implementing the required structured reasoning pipeline with minimal disruption to the codebase.