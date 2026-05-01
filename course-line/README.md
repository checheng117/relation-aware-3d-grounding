# Course-Line: CSC6133 Final Project 95+ System

## Purpose

This directory (`course-line/`) contains all strategic planning, evidence organization, and enhancement materials for the **CSC6133 Final Project** with a 95+ score target.

## Course-Line vs Paper-Line: Critical Distinction

| Aspect | course-line | paper-line |
|--------|-------------|------------|
| **Location** | `course-line/` (this directory) | `writing/paper-line/` |
| **Goal** | Course high score (95+) | Paper acceptance (AAAI/etc) |
| **Audience** | Course instructors, TAs | Peer reviewers, ACs |
| **Evaluation** | Completeness, presentation, evidence hierarchy, reproducibility | Novelty, SOTA claims, technical contribution |
| **Content scope** | Full project journey including pilots, diagnostics, limitations | Only defensible core claims |
| **Tone** | Educational, transparent about process | Assertive, claim-focused |

### Why Separate?

1. **Different scoring criteria**: Course projects reward completeness and honest limitation acknowledgment; papers reward focused novelty claims
2. **Different evidence thresholds**: Course allows "pilot evidence" labeled as such; papers require full held-out validation
3. **Different packaging**: Course needs presentation slides, reproducibility notes, appendix materials; papers need camera-ready PDFs

**Do NOT mix course-line and paper-line materials.** This separation prevents:
- Over-claiming in paper submissions
- Under-packaging for course evaluation
- Confusion about evidence levels

## Current Main Line

**Latent Conditioned Relation Scoring for 3D Visual Grounding**

The course report argues that:
1. 3D visual grounding requires reasoning about ambiguous spatial relations
2. Standard baselines fail systematically on hard cases (clutter, multi-anchor)
3. A latent-conditioned architecture enables multi-mode relation reasoning
4. Controlled experiments show +2.1% gain from architecture
5. Counterfactual and latent-mode pilots show promising directions

## Directory Structure

```
course-line/
├── README.md                        # This file: overview and navigation
├── COURSE_LINE_POSITIONING.md       # Why course-line exists, how it differs from paper-line
├── FINAL_PROJECT_EVIDENCE_MAP.md    # Hierarchical evidence catalog (Level A/B/C/D)
├── COURSE_LINE_TODO.md              # Prioritized enhancement tasks (P0/P1/P2)
├── MAIN_TABLE_PLAN.md               # Unified protocol table design
├── QUAL_CASES_PLAN.md               # Qualitative case study plan
├── PRESENTATION_PLAN.md             # 10-15 min presentation structure
├── CLAIM_BOUNDARY.md                # What can/cannot be claimed, why, how to frame
└── REPRODUCIBILITY_NOTES.md         # Scripts, data sources, reproduction commands
```

## Evidence Levels Used in This Directory

| Level | Description | Suitable for |
|-------|-------------|--------------|
| **A** | Final / trusted evidence | Report main text, main table |
| **B** | Controlled supporting evidence | Report main text, subset tables |
| **C** | Pilot evidence | Appendix, presentation backup, discussion |
| **D** | Pending / future work | Future work section, appendix only |

**Rule**: Never present Level C/D evidence as Level A/B in any course submission.

## Related Directories

| Directory | Purpose |
|-----------|---------|
| `writing/course-line/` | Actual course report source (LaTeX, PDF, assets) |
| `writing/paper-line/` | Paper manuscript source (AAAI format) |
| `reports/` | Diagnostic reports, experiment logs |
| `update/` | Phase summaries, mechanism diagnosis |
| `outputs/` | Raw experiment results (JSON, checkpoints) |
| `scripts/` | Training and evaluation scripts |
| `src/` | Implementation source code |

## Quick Navigation

- **Starting the course report?** → Read `FINAL_PROJECT_EVIDENCE_MAP.md` first
- **Planning enhancements?** → See `COURSE_LINE_TODO.md`
- **Building the main table?** → Follow `MAIN_TABLE_PLAN.md`
- **Preparing presentation?** → Use `PRESENTATION_PLAN.md`
- **Uncertain about claims?** → Check `CLAIM_BOUNDARY.md`
- **Need reproducibility?** → Consult `REPRODUCIBILITY_NOTES.md`

## Last Updated

2026-04-23
