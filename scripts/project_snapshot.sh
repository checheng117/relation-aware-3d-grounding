#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports

OUT="reports/project_snapshot_$(date +%Y%m%d_%H%M).md"

{
  echo "# Project Snapshot"
  echo
  echo "Generated: $(date)"
  echo
  echo "## pwd"
  pwd
  echo
  echo "## Git Status"
  git status --short || true
  echo
  echo "## Top-level Files"
  find . -maxdepth 2 -type f \
    -not -path './.git/*' \
    -not -path './__pycache__/*' \
    | sort | sed -n '1,300p'
  echo
  echo "## Important Candidate Files"
  find . -maxdepth 4 -type f \
    \( -name '*.md' -o -name '*.tex' -o -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' -o -name '*.bib' -o -name '*.sh' \) \
    -not -path './.git/*' \
    | sort | sed -n '1,500p'
} > "$OUT"

echo "Wrote $OUT"
