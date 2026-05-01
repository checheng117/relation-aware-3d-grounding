# Course-Line Positioning Document

## What is Course-Line?

**Course-line** is the strategic planning system for the CSC6133 Final Project submission. It organizes:

1. **Evidence hierarchy** - Which results are final vs pilot vs pending
2. **Enhancement planning** - What additions will maximize course score
3. **Packaging strategy** - How to present work for course evaluation criteria
4. **Claim boundaries** - What can be claimed at what confidence level

### Course-Line Deliverables

- This planning directory (`course-line/`)
- Enhanced course report (`writing/course-line/report.pdf`)
- Presentation slides (10-15 minutes)
- Reproducibility package (scripts, commands, data mappings)

---

## What is Paper-Line?

**Paper-line** is the manuscript preparation system for academic paper submission (e.g., AAAI). It organizes:

1. **Defensible core claims** - Only evidence strong enough for peer review
2. **Minimal publishable unit** - Smallest complete contribution story
3. **Reviewer-facing narrative** - Assertive, novelty-forward framing
4. **Camera-ready packaging** - PDF, supplementary material, rebuttal prep

### Paper-Line Deliverables

- This planning directory (`paper-line/`)
- Paper manuscript (`writing/paper-line/main_draft.tex`)
- Supplementary material
- Rebuttal documents (when needed)

---

## Why Separate Course-Line and Paper-Line?

### 1. Different Evaluation Criteria

| Criterion | Course (95+ target) | Paper (acceptance target) |
|-----------|---------------------|---------------------------|
| **Completeness** | High weight: show full project journey | Medium: only what supports core claim |
| **Novelty** | Medium: demonstrate understanding | High: must be novel contribution |
| **Honesty about limitations** | High weight: shows maturity | Medium: limitations section, but don't overshoot |
| **Presentation quality** | High weight: slides, clarity | Medium: PDF quality, figures |
| **Reproducibility** | High weight: scripts, documentation | Low: code availability nice-to-have |
| **Evidence hierarchy** | Must distinguish final vs pilot | Must all be final/validated |

### 2. Different Audiences

**Course evaluators** want to see:
- You understand the research process
- You can diagnose failures honestly
- You can present work clearly
- You package materials professionally

**Paper reviewers** want to see:
- A novel, defensible claim
- Strong experimental support
- Comparison to relevant baselines
- Clear positioning vs prior work

### 3. Different Risk Tolerance

| Risk type | Course-line | Paper-line |
|-----------|-------------|------------|
| Including pilot results | ✅ OK if labeled | ❌ Never |
| Discussing failures | ✅ Shows maturity | ⚠️ Only in limitations |
| Multiple ablations | ✅ Shows thoroughness | ⚠️ Only if supports claim |
| Future work section | ✅ Expected | ✅ Expected |
| Unfinished experiments | ⚠️ Label as pending | ❌ Never |

---

## Course Scoring Perspective: What to强化 (Strengthen)

For a 95+ course project score, prioritize:

### P0: Must-Have for 95+

1. **Clear main result table** with unified protocol
   - Shows you understand experimental rigor
   - Instructors can quickly verify core claims

2. **Evidence hierarchy transparency**
   - Explicitly label what's final vs pilot
   - Shows research maturity

3. **At least 2 qualitative cases** (success + failure)
   - Demonstrates deep understanding
   - Makes presentation memorable

4. **Complete reproducibility notes**
   - Scripts mapped to results
   - Commands to reproduce each table/figure

5. **Professional presentation** (10-15 min)
   - Clear slides with single message per slide
   - Good figures, readable tables

### P1: Should-Have for 95+

6. **Hard subset analysis**
   - Shows you understand where method works/fails
   - Adds depth beyond overall accuracy

7. **Mechanism explanation**
   - Why does the method work?
   - Architecture diagram, ablation studies

8. **Appendix with extended materials**
   - Extended tables
   - Additional diagnostic figures
   - Training curves

### P2: Nice-to-Have

9. **Multi-seed results**
   - Shows stability awareness
   - Not required for 95+ if single-seed is honest

10. **Statistical significance tests**
    - Sophisticated but optional
    - Can be mentioned as future work

11. **Comparison to additional baselines**
    - SAT or other models if available
    - Not required if ReferIt3DNet baseline is solid

---

## What Course-Line Does NOT Do

1. **Does NOT replace paper-line**
   - Paper-line continues separately for AAAI submission
   - Different audiences, different goals

2. **Does NOT lower standards**
   - Course-line still requires intellectual honesty
   - Pilots must be labeled as pilots

3. **Does NOT mean over-claiming**
   - Course evaluators appreciate honesty about limitations
   - Better to under-claim and be credible

4. **Does NOT create new experiments from scratch**
   - Uses existing repository materials
   - Organizes, labels, and packages them properly

---

## Summary: The Relationship

```
┌─────────────────────────────────────────────────────────────┐
│                     This Repository                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  course-line/          paper-line/                           │
│  ├─ For CSC6133        ├─ For AAAI submission                │
│  ├─ 95+ target         ├─ Acceptance target                  │
│  ├─ Complete journey   ├─ Minimal publishable unit           │
│  ├─ Pilots allowed*    ├─ Only final evidence                │
│  └─ Emphasis on:       └─ Emphasis on:                       │
│     - Presentation        - Novelty                           │
│     - Reproducibility     - Defensibility                     │
│     - Completeness        - SOTA comparison                   │
│     - Honesty             - Focused claim                     │
│                                                                  │
└─────────────────────────────────────────────────────────────┘

* Pilots must be explicitly labeled as Level C evidence
```

**Key principle**: Course-line and paper-line share the same underlying research, but package it differently for different audiences and evaluation criteria. Neither is "easier" — they optimize for different goals.
