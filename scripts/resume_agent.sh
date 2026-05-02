#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " Research Agent Resume"
echo "============================================================"
echo

echo "## Project Root"
pwd
echo

echo "## Git Branch"
git branch --show-current 2>/dev/null || echo "No git branch found"
echo

echo "## Git Status"
git status --short 2>/dev/null || echo "No git repository found"
echo

echo "## Agent Entry Files"

files=(
  ".agent/START_HERE.md"
  ".agent/00_state/PROJECT_STATE.md"
  ".agent/00_state/SESSION_STATE.md"
  ".agent/00_state/NEXT_ACTION.md"
  ".agent/00_state/CLAIM_LEDGER.md"
  ".agent/00_state/EXPERIMENT_LEDGER.md"
  ".agent/00_state/TODO.md"
)

for f in "${files[@]}"; do
  if [ -f "$f" ]; then
    echo "OK   $f"
  else
    echo "MISS $f"
  fi
done

echo

echo "## Latest Plan"
latest_plan="$(ls -t .agent/10_plans/*.md 2>/dev/null | head -1 || true)"
if [ -n "${latest_plan:-}" ]; then
  echo "$latest_plan"
  echo
  sed -n '1,80p' "$latest_plan"
else
  echo "No plan found under .agent/10_plans/"
fi

echo

echo "## Latest Execution Report"
latest_exec="$(ls -t .agent/20_exec/*.md 2>/dev/null | head -1 || true)"
if [ -n "${latest_exec:-}" ]; then
  echo "$latest_exec"
  echo
  sed -n '1,100p' "$latest_exec"
else
  echo "No execution report found under .agent/20_exec/"
fi

echo

echo "## Latest Verification Report"
latest_verify="$(find .agent/30_verify -maxdepth 1 -type f -iname '*verify*.md' 2>/dev/null | sort | tail -1 || true)"
if [ -n "${latest_verify:-}" ]; then
  echo "$latest_verify"
  echo
  sed -n '1,100p' "$latest_verify"
else
  echo "No verification report found under .agent/30_verify/"
fi

echo

echo "## Diff Stat"
git diff --stat 2>/dev/null || true

echo
echo "============================================================"
echo " Next Recommended Action"
echo "============================================================"
echo

if [ -n "$(git status --short 2>/dev/null || true)" ]; then
  echo "There are uncommitted changes."
  echo
  echo "Recommended:"
  echo "  1. Inspect:"
  echo "       git status --short"
  echo
  echo "  2. If tracked files were accidentally deleted, run:"
  echo "       git restore -- \$(git diff --name-only --diff-filter=D)"
  echo
  echo "  3. Generate verifier packet:"
  echo "       ./scripts/make_verify_packet.sh"
  echo
  echo "  4. Ask Antigravity/Gemini/Codex to verify before commit."
else
  echo "No uncommitted changes detected."
  echo
  echo "Recommended:"
  echo "  claude"
  echo "  /create-plan \"create the next concrete plan based on current PROJECT_STATE, SESSION_STATE, and TODO\""
fi

echo
echo "If lost, read:"
echo "  .agent/START_HERE.md"
echo "  .agent/00_state/NEXT_ACTION.md"
echo
