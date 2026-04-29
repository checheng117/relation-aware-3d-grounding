#!/usr/bin/env bash
set -euo pipefail

mkdir -p .agent/30_verify

OUT=".agent/30_verify/VERIFY_PACKET.md"

{
  echo "# VERIFY PACKET"
  echo
  echo "Generated: $(date)"
  echo
  echo "## Branch"
  git branch --show-current || true
  echo
  echo "## Git Status"
  git status --short || true
  echo
  echo "## Diff Stat"
  git diff --stat || true
  echo
  echo "## Changed Files"
  git diff --name-only || true
  echo
  echo "## Latest Plan"
  latest_plan=$(ls -t .agent/10_plans/*.md 2>/dev/null | head -1 || true)
  if [ -n "${latest_plan:-}" ]; then
    echo "Plan file: $latest_plan"
    echo
    sed -n '1,260p' "$latest_plan"
  else
    echo "No plan found."
  fi
  echo
  echo "## Latest Execution Report"
  latest_exec=$(ls -t .agent/20_exec/*.md 2>/dev/null | head -1 || true)
  if [ -n "${latest_exec:-}" ]; then
    echo "Execution report: $latest_exec"
    echo
    sed -n '1,260p' "$latest_exec"
  else
    echo "No execution report found."
  fi
  echo
  echo "## Current Diff"
  git diff -- .agent paper src scripts prompts README.md requirements.txt pyproject.toml setup.py Makefile 2>/dev/null | sed -n '1,1600p'
} > "$OUT"

echo "Wrote $OUT"
wc -l "$OUT"
