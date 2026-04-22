# Release Commit Plan

**Date**: 2026-04-22
**Status**: READY FOR USER EXECUTION

---

## Overview

This document provides step-by-step instructions for committing and pushing the open-source release to GitHub.

**Prerequisites**:
- GitHub SSH key configured
- All audit checks reviewed and accepted
- `.env` file confirmed deleted

---

## Step 1: Configure SSH Remote (Required)

The repository currently uses HTTPS. Switch to SSH:

```bash
cd /home/cc/Project/CC/relation-aware-3d-grounding
git remote set-url origin git@github.com:checheng117/relation-aware-3d-grounding.git
```

Verify the change:

```bash
git remote -v
# Should show:
# origin	git@github.com:checheng117/relation-aware-3d-grounding.git (fetch)
# origin	git@github.com:checheng117/relation-aware-3d-grounding.git (push)
```

Test SSH connection:

```bash
ssh -T git@github.com
# Should show: "Hi checheng117! You've successfully authenticated..."
```

---

## Step 2: Review Changes

Before staging, review what will be committed:

```bash
# See current status
git status

# Review deleted files (old configs, scripts, reports)
git status --short | grep "^D"

# Review new untracked files
git status --short | grep "^?"
```

**Deleted files summary** (intentional cleanup):
- Old experiment configs (~70 files in configs/)
- Old reports from earlier phases (~50 files)
- Old scripts (~50 files)
- These are archived in git history if needed later

**New files to add** (release artifacts):
- `reports/open_source_release_audit.md`
- `reports/open_source_release_checklist.md`
- `reports/open_source_release_boundary.md`
- `reports/method_freeze_and_release_policy.md`
- `reports/diagnostic_paper_positioning_freeze.md`
- `reports/final_diagnostic_master_summary.md`
- `reports/final_diagnostic_master_table.csv`
- `reports/dense_fundamentals_summary.md`
- `reports/readme_open_source_alignment.md`
- `README.md` (updated for release)
- Core model code in `src/rag3d/models/`
- Retained scripts in `scripts/`
- Retained configs in `configs/cover3d_round1/` and `configs/cover3d_smoke.yaml`

---

## Step 3: Stage Changes

Stage files in logical groups:

```bash
# 1. Add new reports
git add reports/open_source_release_audit.md
git add reports/open_source_release_checklist.md
git add reports/open_source_release_boundary.md
git add reports/method_freeze_and_release_policy.md
git add reports/diagnostic_paper_positioning_freeze.md
git add reports/final_diagnostic_master_summary.md
git add reports/final_diagnostic_master_table.csv
git add reports/dense_fundamentals_summary.md
git add reports/readme_open_source_alignment.md

# 2. Add updated README
git add README.md

# 3. Add .claude/ files (for local dev tracking, excluded from git by .gitignore)
# Note: .claude/ is now in .gitignore, so these won't be tracked

# 4. Add core model code
git add src/rag3d/models/

# 5. Add retained scripts
git add scripts/train_cover3d_round1.py
git add scripts/analyze_dense_fundamentals.py
git add scripts/check_clean_baseline_readiness.py

# 6. Add retained configs
git add configs/cover3d_smoke.yaml
git add configs/dataset/expanded_nr3d.yaml
git add configs/sat_baseline.yaml

# 7. Add repro baseline configs
git add repro/referit3d_baseline/configs/
git add repro/sat_baseline/

# 8. Add new diagnostic scripts
git add src/rag3d/parsers/span_alignment.py

# 9. Stage deleted files (cleanup)
git add -u
```

Verify staged changes:

```bash
git status
```

---

## Step 4: Create Release Commit

Create the commit with a comprehensive message:

```bash
git commit -m "$(cat <<'EOF'
Open-source release: Diagnostic paper companion v1.0

This release establishes the repository as a diagnostic paper and benchmark
companion, not a strong method contribution.

## What This Release Contains

### Core Evaluation Foundation
- Recovered full Nr3D dataset (41,503 samples, 641 ScanNet scenes)
- Scene-disjoint train/val/test splits (verified zero overlap)
- Reproduced baselines: ReferIt3DNet (30.79%), SAT (28.27%)

### Method Results
- Dense-no-cal-v1: 31.05% Acc@1 (+0.22% over baseline)
- Modest but real gain, lightweight (398K parameters)
- Multi-anchor subsets: +5.3% improvement

### Diagnostic Findings
- Calibration line: FROZEN (signals uninformative, gate collapse)
- Dense strengthening line: FROZEN (relation scores mostly noisy)
- Score gap analysis: -0.95 (correct targets score LOWER)
- Pair ranking Hit@1: 7.0% (weak anchor selection)

### Documentation
- Complete diagnostic reports with evidence
- Method freeze policy with restart prerequisites
- Open-source release boundary documentation
- Reproduction scripts and configs

## What This Release Does NOT Contain

- SOTA-challenging results (modest +0.22% gain)
- Active calibration development (line is frozen)
- Dense scorer enhancements (line is frozen)
- Multi-seed validation (single-seed only)

## Target Venues

Primary: TACL, ACL Findings, EMNLP Findings, Scientific Data
Secondary: CVPR/ICCV/ECCV Workshop, 3DV, BMVC
Not for: AAAI/NeurIPS/ICML/CVPR main track (method signal insufficient)

## Release Artifacts

New files:
- reports/open_source_release_audit.md
- reports/open_source_release_checklist.md
- reports/open_source_release_boundary.md
- reports/method_freeze_and_release_policy.md
- reports/diagnostic_paper_positioning_freeze.md
- reports/final_diagnostic_master_summary.md
- README.md (rewritten for open-source audience)

Cleanup:
- Archived ~170 old experiment configs
- Archived ~50 old phase reports
- Archived ~50 debug/experiment scripts
- Removed .env with credentials (privacy fix)

## Verification

All audit checks pass:
- Documentation consistency: PASS
- Material completeness: PASS
- Privacy/sensitive info: PASS (credentials removed)
- Repository publishability: PASS
- Reproduction paths: PASS
- Git preparation: PASS

Tag: v1.0-diagnostic-release
EOF
)"
```

---

## Step 5: Push to GitHub

Push the release commit:

```bash
git push origin main
```

Expected output:
```
Enumerating objects: XXX, done.
Counting objects: 100% (XXX/XXX), done.
Delta compression using up to XX threads
Compressing objects: 100% (XXX/XXX), done.
Writing objects: 100% (XXX/XXX), XXX KiB | X.XX MiB/s, done.
Total XXX (delta XX), reused XX (delta XX), pack-reused XX
remote: Resolving deltas: 100% (XX/XX), done.
To github.com:checheng117/relation-aware-3d-grounding.git
   abc1234..def5678  main -> main
```

---

## Step 6: Create Release Tag (Optional)

Create a git tag for the release:

```bash
git tag -a v1.0-diagnostic-release -m "Diagnostic paper companion release v1.0"
git push origin v1.0-diagnostic-release
```

---

## Step 7: Post-Push Verification

After push completes:

1. Visit repository on GitHub: https://github.com/checheng117/relation-aware-3d-grounding
2. Verify commit appears in history
3. Verify README renders correctly
4. Verify LICENSE is visible
5. Check that no `.env` or sensitive files are visible

---

## Rollback Plan (If Needed)

If issues are discovered after push:

```bash
# 1. Fix the issue locally
# 2. Create a fix commit
git commit -m "Fix: <description>"

# 3. Push the fix
git push origin main

# For serious issues requiring history rewrite (rare):
# git reset --hard HEAD~1
# git push --force origin main
# (Only if no one else has pulled the release)
```

---

## Summary Checklist

- [ ] SSH remote configured
- [ ] Changes reviewed with `git status`
- [ ] Files staged with `git add`
- [ ] Release commit created
- [ ] Pushed to GitHub
- [ ] Release tag created (optional)
- [ ] Repository verified on GitHub

---

## Emergency Contacts

If SSH fails:
- Check SSH key: `cat ~/.ssh/id_ed25519.pub`
- Add to GitHub: https://github.com/settings/keys
- Test connection: `ssh -T git@github.com`

If push fails:
- Check branch protection rules on GitHub
- Verify no merge conflicts
- Run `git fetch origin` and rebase if needed

---

**End of Release Commit Plan**
