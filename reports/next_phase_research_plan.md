# Next Phase Research Plan

**Date**: 2026-04-19
**Previous Phase**: Experimentation (CLOSED)
**Current Phase**: Project Delivery & Analysis

---

## Phase Transition

| Previous Phase | Status |
|----------------|--------|
| Baseline reproduction | COMPLETE (30.79% trusted) |
| Secondary baseline (SAT) | COMPLETE (28.27% verified) |
| Custom method exploration | CLOSED (blocked by hardware) |

**No more experiments on current hardware.**

---

## Priority Order (Strict)

### A. Project Delivery (Highest Priority)

| Task | Description | Timeline |
|------|-------------|----------|
| A1 | Final report document | 1-2 days |
| A2 | Presentation slides | 1 day |
| A3 | Code cleanup and documentation | 1 day |
| A4 | Results summary tables | Done |

**This phase must complete before any further research.**

### B. Baseline Analysis (High Priority)

| Task | Description | Value |
|------|-------------|-------|
| B1 | Error analysis on ReferIt3DNet | Identify failure patterns |
| B2 | Query type breakdown | Which types are hardest |
| B3 | Spatial relation categories | Map to implicit relation potential |
| B4 | Comparison to public benchmark | Explain gap to 35.6% |

**This informs future method design, no GPU needed.**

### C. Future Method Continuation (Medium Priority, Conditional)

| Task | Requirement | Description |
|------|-------------|-------------|
| C1 | Stable hardware | Resume v3 training epochs 16-30 |
| C2 | Different machine | Complete v3 validation |
| C3 | Full evaluation | Test Acc@1 comparison to baseline |

**Only when stable GPU machine is available.**

### D. New Methods (Low Priority, Future)

| Method | When to Consider |
|--------|------------------|
| Attention pooling | After v3 validated |
| Graph neural networks | After v3 validated |
| Multi-modal fusion | After baseline analysis |

**Only after A, B, and C complete.**

---

## No Random Experimentation

| Forbidden Action | Reason |
|------------------|--------|
| Hyperparameter sweeps | Hardware unstable |
| New architecture trials | Need validated baseline first |
| Training on current GPU | Confirmed driver crash |

---

## Recommended Immediate Actions

### Day 1-2: Project Delivery

1. Write final report summarizing all findings
2. Create presentation with:
   - Baseline reproduction results
   - Custom method exploration summary
   - v3 promising signal (unconfirmed)
   - Hardware blocker explanation
   - Future continuation path

### Day 3: Baseline Analysis

1. Analyze ReferIt3DNet test failures
2. Categorize by query type
3. Identify spatial relation patterns
4. Document findings

### Future (when stable hardware available):

1. Resume v3 from checkpoint
2. Complete remaining epochs
3. Evaluate on test split
4. Publish final comparison

---

## Project Deliverables Checklist

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Baseline reproduction | DONE | 30.79% trusted |
| SAT reproduction | DONE | 28.27% verified |
| Parser methods | DONE | Discarded |
| Implicit v1/v2 | DONE | Documented |
| Implicit v3 | ARCHIVED | Promising, hardware blocked |
| Final report | TODO | Priority A |
| Presentation | TODO | Priority A |
| Error analysis | TODO | Priority B |

---

## Success Criteria for Phase Closure

| Criterion | Status |
|-----------|--------|
| All experiments documented | DONE |
| Trusted baseline established | DONE (30.79%) |
| Hardware blocker identified | DONE |
| Future path defined | DONE |
| No pending experiments | DONE |

---

## Final Statement

**Current phase is closed.**

**Next immediate action**: Project delivery (report, presentation).

**No experiments will be run on current hardware.**

**Future continuation**: Resume v3 only on stable GPU machine.