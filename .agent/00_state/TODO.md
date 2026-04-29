# TODO

## P0: Must Do Next
- [x] Run bootstrap project scan. (DONE: `20260429_1430_bootstrap_project_scan.md`)
- [x] Identify project goal, main claim, current evidence, and missing evidence. (DONE)
- [ ] Create first executable plan for next priority action.

## P1: Important (High Priority)
- [ ] **Configure headless/remote environment for full training**
  - Blocker for E-011, E-012 (10-epoch validation)
  - See `update/LOCAL_CRASH_DIAGNOSIS.md` for local limitations
- [ ] Run full E0-matched and E1-CF training (10 epochs, batch=64)
- [ ] Finalize paper structure with honest claim framing
  - Follow `course-line/CLAIM_BOUNDARY.md` for evidence boundaries
  - Use wording from `update/PHASE5_CONCLUSION_AND_PAPER_NOTES.md`

## P2: Medium Priority
- [ ] Complete MoE full training (K=1 vs K=4, 80 epochs) - if compute available
- [ ] Multi-seed validation for statistical significance
- [ ] Build paper/code skeleton map
- [ ] Create final paper outline based on evidence hierarchy

## P3: Optional / Future Work
- [ ] Metadata pipeline enhancement (enable RHN ablation)
- [ ] Anchor-chain reasoning (deferred - no current gain)
- [ ] Multi-seed validation
- [ ] Statistical significance tests

## Completed
- [x] Bootstrap project scan complete
- [x] Project goal identified: Latent Conditioned Relation Scoring
- [x] Main claim identified: Architecture +2.1%, supervision not needed
- [x] Evidence mapped by level (A/B/C/D)
- [x] Missing evidence identified: full training, multi-seed, RHN
- [x] Important files/scripts catalogued
- [x] Risks documented

## Current Blockers
1. **Full training requires headless environment** - local machine crashes
2. **RHN ablation impossible** - missing object metadata in embeddings
3. **Multi-anchor shows no improvement** - cannot claim solved

## Immediate Next Steps
1. Decide: Proceed with paper using pilot evidence, OR wait for full training
2. If proceeding: Draft paper with honest limitations section
3. If waiting: Configure remote GPU/cloud instance for full training