# Phase 3 Parser Integration Audit

**Date**: 2026-04-02
**Author**: Claude Code audit

---

## 1. Current Parser Interfaces

### 1.1 Existing Parser Classes

| File | Class | Status | Notes |
|---|---|---|---|
| `src/rag3d/parsers/base.py` | `BaseParser` | Active | Abstract base with `parse(raw_text) -> ParsedUtterance` |
| `src/rag3d/parsers/heuristic_parser.py` | `HeuristicParser` | Active | Regex-based, finds relations via `_REL_PATTERNS`, estimates confidence |
| `src/rag3d/parsers/structured_rule_parser.py` | `StructuredRuleParser` | Active | Template-first, single relation, different confidence calibration |
| `src/rag3d/parsers/cached_parser.py` | `CachedParser` | Active | Wraps any `BaseParser` with SHA256-keyed disk cache |
| `src/rag3d/parsers/hf_structured_parser.py` | `HFStructuredParser` | Stub | Returns heuristic fallback unless env var set; no real VLM call |
| `src/rag3d/parsers/parser_cache.py` | `HeuristicStructuredParser` | Duplicate | Similar to HeuristicParser but with different keyword list; **redundant** |
| `src/rag3d/parsers/parser_cache.py` | `ParserCache` | Active | Old cache system, uses different key format; **legacy** |
| `src/rag3d/parsers/structured_parser.py` | `StructuredParserInterface` | Wrapper | Calls `get_parsed_utterance()` from parser_cache.py; **legacy path** |

### 1.2 Parser Registry in `__init__.py`

```python
__all__ = ["BaseParser", "CachedParser", "HeuristicParser", "StructuredRuleParser"]
```

Missing: `HFStructuredParser`, any VLM adapter.

### 1.3 ParsedUtterance Schema (src/rag3d/datasets/schemas.py)

**Current fields**:
- `raw_text: str`
- `target_head: str | None`
- `target_modifiers: list[str]`
- `anchor_head: str | None`
- `relation_types: list[str]`
- `parser_confidence: float = 0.5`
- `paraphrase_set: list[str]`
- `parse_source: str = "unknown"`
- `parse_warnings: list[str]`

**Missing fields for Phase 3**:
- `parse_status: str` (valid/partial/invalid/missing/fallback_used)
- `fallback_triggered: bool`
- `fallback_reason: str | None`
- `vlm_metadata: dict | None` (for VLM-specific info like model version, latency)

---

## 2. What is Missing for Real Cached VLM Parser Support

### 2.1 VLM Parser Cache Infrastructure

**Current state**:
- `data/parser_cache/` exists but contains only `.gitkeep`
- `CachedParser` uses SHA256 hash of raw_text as key, single flat directory
- No separation between heuristic and VLM caches

**Required**:
- Separate directories: `data/parser_cache/heuristic/`, `data/parser_cache/vlm/`
- Source-aware cache key or directory structure
- JSON schema for VLM parse records (target_head, anchor_head, relation_types, confidence, model_version)

### 2.2 VLM Parser Adapter

**Required**:
- `VlmParserAdapter(BaseParser)` that:
  - Loads pre-cached VLM parses from `data/parser_cache/vlm/*.json`
  - Validates JSON fields before returning `ParsedUtterance`
  - Returns parse_status = "missing" if cache file not found
  - Returns parse_status = "invalid" if JSON malformed
  - Does NOT call VLM online (offline-only)

### 2.3 Parse Quality Validation

**Required**:
- Function to validate parse quality:
  - Check `target_head` present and non-empty
  - Check `anchor_head` present when `relation_types` non-empty
  - Check `parser_confidence` in valid range [0, 1]
  - Return `parse_status` string: "valid" / "partial" / "invalid"

### 2.4 Fallback Controller

**Required**:
- `FallbackController` module that decides:
  - Given `ParsedUtterance` + confidence threshold
  - Return: should_fallback (bool), fallback_mode (none/hard/hybrid)
  - For hybrid: return structured_weight, raw_text_weight

---

## 3. Where Fallback Should Be Inserted

### 3.1 Primary Insertion Point: StructuredRelationModel.forward()

**Current flow**:
```python
def forward(batch, parsed_list=None):
    # Uses parsed_list if provided, else falls back to raw h_t
    if parsed_list is not None:
        anchor_query = self._get_anchor_query_from_parsed(parsed_list, h_t)
    else:
        anchor_query = h_t  # raw text embedding
```

**Problem**: No controlled fallback logic. If parse is bad, still tries to use it.

**Required modification**:
```python
def forward(batch, parsed_list=None, fallback_controller=None):
    # For each sample, decide fallback based on parse quality
    # If fallback triggered, use raw_text_relation scoring path
    # Export fallback decision in output dict
```

### 3.2 Training Runner: forward_relation_aware()

**Current (src/rag3d/training/runner.py:223)**:
```python
def forward_relation_aware(model, batch, parser):
    samples = batch["samples_ref"]
    parsed_list = [parser.parse(s.utterance) for s in samples]
    logits, _ = model(batch, parsed_list=parsed_list)
    return logits
```

**Required modification**:
- Pass `fallback_controller` to forward function
- Track fallback statistics during training

### 3.3 Eval Script: run_structured_relation_model.py

**Current**: No parser integration in forward pass (passes `parsed_list=None`).

**Required modification**:
- Load parser based on config (heuristic / cached_vlm)
- Pass parsed_list to model
- Export parser_source, parse_status, fallback_triggered per prediction

---

## 4. Minimal-Intrusion Implementation Plan

### Step 1: Extend ParsedUtterance Schema

Add fields:
```python
parse_status: str = "unknown"  # valid/partial/invalid/missing/fallback_used
fallback_triggered: bool = False
fallback_reason: str | None = None
vlm_metadata: dict | None = None
```

### Step 2: Create Parse Quality Validator

File: `src/rag3d/parsers/parse_quality.py`

```python
def validate_parse_quality(parsed: ParsedUtterance) -> str:
    # Returns parse_status string
```

### Step 3: Create VlmParserAdapter

File: `src/rag3d/parsers/vlm_parser_adapter.py`

```python
class VlmParserAdapter(BaseParser):
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir / "vlm"

    def parse(self, raw_text: str) -> ParsedUtterance:
        # Load from cache, validate, return with parse_status
```

### Step 4: Create FallbackController

File: `src/rag3d/relation_reasoner/fallback_controller.py`

```python
class FallbackController:
    def __init__(self, mode: str, confidence_threshold: float):
        # mode: "none", "hard", "hybrid"

    def should_fallback(self, parsed: ParsedUtterance) -> tuple[bool, str, float, float]:
        # Returns (fallback, reason, structured_weight, raw_text_weight)
```

### Step 5: Extend StructuredRelationModel

Modify `forward()` to:
- Accept `fallback_controller` parameter
- Compute per-sample fallback decision
- Use hybrid scoring when fallback triggered
- Export fallback info in return dict

### Step 6: Add Config Fields

In `configs/train/structured_relation_model.yaml`:
```yaml
parser_source: heuristic  # heuristic / cached_vlm
fallback_mode: none       # none / hard / hybrid
fallback_confidence_threshold: 0.5
```

### Step 7: Create Phase 3 Scripts

- `scripts/run_vlm_parser_structured_model.py`
- `scripts/run_phase3_parser_ablation.py`

### Step 8: Create VLM Cache Directory Structure

```
data/parser_cache/
  heuristic/
    *.json  (existing CachedParser outputs)
  vlm/
    *.json  (pre-generated VLM parses)
```

---

## 5. Design Constraints Checklist

| Constraint | Current Status | Action |
|---|---|---|
| Minimal intrusion | Multiple legacy parser paths exist | Deprecate parser_cache.py legacy, use CachedParser + BaseParser |
| Parser source distinguishable | parse_source field exists | Ensure VLM adapter sets parse_source="cached_vlm" |
| Fallback explicit | No fallback logic exists | Add FallbackController, export per-prediction |
| No uncontrolled sweeps | Config-driven | Add parser_source/fallback_mode to config |
| No fake VLM online dep | HFStructuredParser is stub | VlmParserAdapter loads offline cache only |

---

## 6. Recommended Execution Order

1. Extend `ParsedUtterance` schema (minimal change)
2. Create `parse_quality.py` validator
3. Create `VlmParserAdapter` (offline-only)
4. Create `FallbackController`
5. Extend `StructuredRelationModel.forward()` with fallback support
6. Add config fields for parser_source/fallback_mode
7. Create experiment scripts and configs
8. Run ablations and collect results

---

## 7. Key Files to Modify/Add

| Action | File |
|---|---|
| Extend | `src/rag3d/datasets/schemas.py` |
| Add | `src/rag3d/parsers/parse_quality.py` |
| Add | `src/rag3d/parsers/vlm_parser_adapter.py` |
| Add | `src/rag3d/relation_reasoner/fallback_controller.py` |
| Extend | `src/rag3d/relation_reasoner/structured_relation_model.py` |
| Extend | `src/rag3d/parsers/__init__.py` |
| Extend | `configs/train/structured_relation_model.yaml` |
| Add | `configs/train/structured_relation_vlm_parser.yaml` |
| Add | `configs/train/structured_relation_vlm_fallback.yaml` |
| Add | `scripts/run_phase3_parser_ablation.py` |

---

## 8. Summary

**Current parser interfaces**:
- `BaseParser`, `HeuristicParser`, `StructuredRuleParser`, `CachedParser` are usable
- `HFStructuredParser` is stub (returns heuristic)
- Legacy `parser_cache.py` code is redundant with `CachedParser`

**Missing for VLM support**:
- VLM parser cache directory separation
- Parse quality validation function
- Fallback controller module
- Extended `ParsedUtterance` schema with parse_status/fallback fields
- Config fields for parser_source and fallback_mode

**Insertion points**:
- `StructuredRelationModel.forward()` for fallback logic
- `forward_relation_aware()` in runner for training
- Eval scripts for per-sample parser integration

**Minimal intrusion approach**:
- Extend existing classes rather than replace
- Deprecate legacy paths but don't delete
- All new logic config-driven
- Offline-only VLM cache, no live API calls