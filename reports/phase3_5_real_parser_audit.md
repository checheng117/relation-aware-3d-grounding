# Phase 3.5 Real Parser Audit

**Date**: 2026-04-02
**Phase**: Real VLM Parser Cache Generation + Parse QA + Formal Rerun

---

## 1. What Currently Supports Real VLM Cache Generation

### 1.1 VlmParserAdapter (src/rag3d/parsers/vlm_parser_adapter.py)

**Status**: ✅ Ready for loading real cached parses

- Loads from `data/parser_cache/vlm/*.json`
- Uses SHA256 hash of `raw_text` as cache key
- Validates JSON against `ParsedUtterance` schema
- Sets `parse_source="cached_vlm"` automatically
- Returns `parse_status="missing"` for uncached utterances
- Supports `cache_parse()` method for writing new cache entries

**Key method for cache population**:
```python
def cache_parse(self, raw_text: str, parsed: ParsedUtterance) -> None:
    """Save a parse to VLM cache (for pre-generation scripts)."""
```

### 1.2 ParsedUtterance Schema (src/rag3d/datasets/schemas.py)

**Status**: ✅ Ready

Fields supported:
- `raw_text: str`
- `target_head: str | None`
- `target_modifiers: list[str]`
- `anchor_head: str | None`
- `relation_types: list[str]`
- `parser_confidence: float`
- `paraphrase_set: list[str]`
- `parse_source: str`
- `parse_warnings: list[str]`
- `parse_status: str` (Phase 3 addition)
- `fallback_triggered: bool` (Phase 3 addition)
- `fallback_reason: str | None` (Phase 3 addition)
- `vlm_metadata: dict[str, Any] | None` (Phase 3 addition)

### 1.3 Parse Quality Validation (src/rag3d/parsers/parse_quality.py)

**Status**: ✅ Ready

- `validate_parse_quality()` → returns "valid", "partial", or "invalid"
- `classify_parse_quality_batch()` → counts by status
- `compute_parse_confidence_bucket()` → "high", "medium", "low"

### 1.4 FallbackController (src/rag3d/relation_reasoner/fallback_controller.py)

**Status**: ✅ Ready

- Modes: "none", "hard", "hybrid"
- Exports fallback statistics
- Integrates with `StructuredRelationModel`

### 1.5 Experiment Runner (scripts/run_phase3_parser_ablation.py)

**Status**: ✅ Ready for real VLM cache

- `build_parser_from_config("cached_vlm")` works
- Exports parse_status, parser_confidence per prediction
- Exports fallback statistics
- Supports controlled conditions

---

## 2. What is Still Mock-Only

### 2.1 generate_mock_vlm_cache.py

**Status**: ❌ Mock only, not suitable for real cache generation

**Problems**:
1. Generates synthetic utterances from templates, not actual dataset utterances
2. Does not read from `val_manifest.jsonl` or `train_manifest.jsonl`
3. Mock parses use simple heuristics, not VLM output
4. No VLM API integration
5. Quality distribution is artificial (70% good, 20% partial, 10% bad)

**Key gap**:
```python
# Current: generates utterances from templates
for template in MOCK_PARSE_TEMPLATES:
    utterance = template.format(target=target, anchor=anchor, ...)

# Needed: read utterances from dataset manifest
manifest = load_manifest("data/processed/val_manifest.jsonl")
for sample in manifest:
    utterance = sample["utterance"]
    # Call real VLM or structured parser
```

### 2.2 VLM Backend

**Status**: ❌ No real VLM backend integrated

**Missing**:
- No API client for Claude, GPT, or local VLM
- No prompt template for structured parse extraction
- No retry/fault-tolerance for API calls
- No token budget management

### 2.3 Cache QA Pipeline

**Status**: ❌ Not implemented

**Missing**:
- Parse coverage measurement
- Target-head quality audit
- Anchor quality audit
- Relation extraction quality audit
- Human-readable sample export
- Parser-to-grounding correlation analysis

---

## 3. Exact Insertion Point for Real Parser Generation

### 3.1 Recommended Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Real Parser Cache Generation                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  val_manifest.jsonl ──► Extract utterances ──► Dedup utterances    │
│                                                                     │
│                          │                                          │
│                          ▼                                          │
│                                                                     │
│               ┌──────────────────────┐                              │
│               │   Parser Backend     │                              │
│               │  (VLM API / Local)   │                              │
│               └──────────┬───────────┘                              │
│                          │                                          │
│                          ▼                                          │
│                                                                     │
│  ParsedUtterance ──► Quality Validation ──► Cache JSON Write       │
│                                                                     │
│                          │                                          │
│                          ▼                                          │
│                                                                     │
│              data/parser_cache/vlm/<hash>.json                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Cleanest Insertion Point

**Create new script**: `scripts/generate_real_vlm_cache.py`

**Reuse**: `VlmParserAdapter.cache_parse()` method already exists

**Flow**:
1. Load utterances from manifest files
2. Deduplicate utterances
3. Check which utterances already have cached parses
4. For missing utterances, call parser backend
5. Validate parse quality
6. Write to cache via `VlmParserAdapter.cache_parse()`
7. Export generation logs and QA statistics

### 3.3 Parser Backend Options

**Option A: Claude/GPT API (recommended for quality)**
- Use Anthropic Claude API with structured JSON output
- Prompt template for parse extraction
- Cost: ~$0.003-0.01 per utterance
- 156 val samples = ~$1-2

**Option B: Local Heuristic Parser (for ablation)**
- Use existing `HeuristicParser` as "fake VLM" baseline
- Already implemented, zero cost
- Good for controlled comparison

**Option C: Local LLM with structured output**
- Use Ollama or similar with structured output mode
- Free but lower quality than Claude

---

## 4. Existing Experiment Scripts That Can Be Reused Without Redesign

### 4.1 scripts/run_phase3_parser_ablation.py

**Reuse**: ✅ Directly usable once real cache is populated

- Already supports `parser_source="cached_vlm"`
- Already exports all required metrics
- Already compares fallback modes
- Already includes raw-text baseline

### 4.2 configs/train/structured_relation_*.yaml

**Reuse**: ✅ Already have correct fields

- `parser_source`: "heuristic" / "cached_vlm"
- `fallback_mode`: "none" / "hard" / "hybrid"
- `fallback_confidence_threshold`: 0.5

### 4.3 VlmParserAdapter

**Reuse**: ✅ Directly usable

- `parse()` loads from cache
- `cache_parse()` writes to cache

---

## 5. Gap Analysis Summary

| Component | Status | Action Needed |
|---|---|---|
| VlmParserAdapter | ✅ Ready | None |
| ParsedUtterance schema | ✅ Ready | None |
| Parse quality validation | ✅ Ready | None |
| FallbackController | ✅ Ready | None |
| StructuredRelationModel | ✅ Ready | None |
| Experiment runner | ✅ Ready | None |
| Configs | ✅ Ready | None |
| Real VLM cache generation | ❌ Missing | Create new script |
| VLM API backend | ❌ Missing | Integrate or use heuristic fallback |
| Cache QA pipeline | ❌ Missing | Create new script |
| Parser-to-grounding analysis | ❌ Missing | Create new script |

---

## 6. Minimal Implementation Plan

### Step 1: Create `scripts/generate_real_vlm_cache.py`

**Requirements**:
- Read utterances from manifest files
- Deduplicate utterances
- Skip utterances already in cache (resumable)
- Call parser backend (start with heuristic as placeholder)
- Write to cache via `VlmParserAdapter.cache_parse()`
- Export generation logs

### Step 2: Create `scripts/validate_vlm_cache.py`

**Requirements**:
- Check cache coverage (utterances in manifest vs cache)
- Compute parse status distribution
- Export sample audit for manual inspection
- Generate `parser_cache_quality_summary.json/md`

### Step 3: Run cache generation for val split

```bash
python scripts/generate_real_vlm_cache.py \
    --manifest data/processed/val_manifest.jsonl \
    --output-dir data/parser_cache/vlm \
    --parser-backend heuristic  # Start with heuristic, upgrade to VLM
```

### Step 4: Run cache QA

```bash
python scripts/validate_vlm_cache.py \
    --cache-dir data/parser_cache/vlm \
    --manifest data/processed/val_manifest.jsonl
```

### Step 5: Run formal Phase 3.5 experiments

```bash
python scripts/run_phase3_parser_ablation.py \
    --device cuda \
    --output-dir outputs/<timestamp>_phase3_5_formal_rerun
```

---

## 7. Key Decisions Needed

### Decision 1: Parser Backend for Real Cache

**Options**:
1. **Heuristic parser** (recommended for initial validation)
   - Zero cost, fast
   - Establishes pipeline correctness
   - Can upgrade to VLM later

2. **Claude API**
   - Higher quality parses
   - Cost: ~$1-2 for val split
   - Requires API key

**Recommendation**: Start with heuristic parser to validate pipeline, then upgrade to VLM if budget allows.

### Decision 2: Cache Key Strategy

**Current**: SHA256 hash of raw_text

**Alternative**: Include sample_id in cache key

**Recommendation**: Keep SHA256 of raw_text (utterance-level deduplication is correct).

### Decision 3: Missing Parse Handling

**Options**:
1. Return `parse_status="missing"` and skip in experiments
2. Fall back to heuristic parse automatically

**Recommendation**: Return "missing" explicitly. Let fallback controller decide what to do.

---

## 8. Files to Create/Modify

### Create:
- `scripts/generate_real_vlm_cache.py`
- `scripts/validate_vlm_cache.py`
- `src/rag3d/parsers/cache_qa.py` (optional, can be in scripts)

### Modify:
- None required for core infrastructure

---

## 9. Conclusion

**Current state**: Parser-aware infrastructure is complete and ready for real VLM cache. The only missing piece is the cache generation script that reads actual dataset utterances and produces cached parses.

**Fastest path to real evidence**:
1. Create cache generation script using existing `HeuristicParser` as backend
2. Generate cache for val_manifest.jsonl utterances
3. Run Phase 3.5 formal rerun
4. Compare heuristic parser (from cache) vs raw-text baseline

**If results are promising**, upgrade to real VLM API backend and repeat.

**If results are not promising**, the infrastructure is still correct—we've just identified that parser quality is not the bottleneck.