# Round-1 Restart Readiness Assessment

**Date**: 2026-04-21
**Status**: RESTART-READY

---

## Executive Summary

**Conclusion**: Round-1 minimal formal experiments are ready to restart.

**Readiness Status**: `RESTART-READY`

---

## 1. Base Parity Status

| Metric | Clean Baseline | Round-1 Base | Gap | Status |
| --- | ---: | ---: | ---: | ---: |
| Test Acc@1 | 30.83% | 30.83% | 0.00% | PASS |
| Test Acc@5 | 91.87% | 91.87% | 0.00% | PASS |

**Tolerance**: ±0.2%
**Result**: Within tolerance

**Fix Applied**: `scripts/train_cover3d_round1.py:516-517`
- Added `masked_fill(~object_mask, float("-inf"))` before argmax
- Prevents padding artifact (0.0 > negative logits)

---

## 2. Dense Stability Status

| Check | Before | After | Status |
| --- | --- | --- | ---: |
| NaN during forward | YES | NO | PASS |
| Completes training | NO | YES | PASS |
| Valid predictions | NO | YES | PASS |

**Fix Applied**: `src/rag3d/models/cover3d_dense_relation.py:350-354`
- Replaced `-inf` with `0.0` before `0 * -inf` multiplication
- Added `nn.GELU()` for smoother activation
- Added `nn.Tanh()` for bounded output

**Smoke Test Result**:
- Dense-no-cal 2 epochs: Acc@1 = 30.95%, Net recovered = +5

---

## 3. Readiness Matrix

| Experiment | Blocking Issue | Status |
| --- | --- | ---: |
| Base-only | Padding artifact | RESOLVED |
| Dense-no-cal | NaN collapse | RESOLVED |
| Dense-calibrated | Blocked by Dense-no-cal | UNBLOCKED |

---

## 4. Recommended Next Steps

### Priority 1: Run Full Round-1 Experiments

```bash
# Base-only (verification)
python scripts/train_cover3d_round1.py --variant base --epochs 0

# Dense-no-cal (full training)
python scripts/train_cover3d_round1.py --variant dense-no-cal --epochs 10

# Dense-calibrated (full training)
python scripts/train_cover3d_round1.py --variant dense-calibrated --epochs 10
```

### Priority 2: Generate Round-1 Results Report

After all three variants complete:
- Compare Acc@1 / Acc@5
- Analyze harmed/recovered cases
- Evaluate hard subset metrics

### Priority 3: Answer Research Questions

**Q1**: Does learned dense scorer convert P2's recoverable conditions into gains?
- Compare Dense-no-cal vs Base on hard subsets

**Q2**: Does calibration reduce harm without suppressing recovery?
- Compare Dense-calibrated vs Dense-no-cal harmed/recovered ratio

---

## 5. Boundaries and Constraints

**Still Frozen**:
- No multi-seed experiments (single seed only)
- No parser/LLM/VLM expansion
- No Apr9/Apr10 logits recovery
- No large-scale method matrix

**Allowed**:
- Full Round-1 (Base / Dense-no-cal / Dense-calibrated)
- Hard subset analysis
- Casebook generation

---

## 6. Files Summary

### Reports Generated

| File | Purpose |
| --- | --- |
| `reports/formal_round1_base_parity_audit.md` | Base parity root cause analysis |
| `reports/dense_stability_repair.md` | Dense NaN fix documentation |
| `reports/round1_restart_readiness.md` | This readiness assessment |

### Code Fixes

| File | Line | Change |
| --- | --- | --- |
| `scripts/train_cover3d_round1.py` | 516-517 | Padding mask for base_pred |
| `src/rag3d/models/cover3d_dense_relation.py` | 98-102 | GELU + Tanh for stability |
| `src/rag3d/models/cover3d_dense_relation.py` | 350-354 | 0*-inf NaN fix |

---

## 7. Final Recommendation

**Recommendation**: Proceed with Round-1 minimal formal experiments.

**Rationale**:
1. Base parity restored (30.83% ± 0.2%)
2. Dense-no-cal stability fixed (no NaN)
3. All blocking issues resolved
4. Infrastructure ready for formal validation

**Risk**: Low. Fixes are minimal and targeted.

**Next Action**: Run full Round-1 training for all three variants.

---

## Appendix: Verification Commands

```bash
# Verify Base parity
python scripts/train_cover3d_round1.py --variant base --epochs 0
# Expected: Acc@1 = 30.83%

# Verify Dense-no-cal stability
python scripts/train_cover3d_round1.py --variant dense-no-cal --epochs 2
# Expected: No NaN warnings, completes successfully

# Verify readiness
cat outputs/cover3d_round1/base_results.json | jq '.acc_at_1'
cat outputs/cover3d_round1/dense-no-cal_results.json | jq '.acc_at_1'
```
