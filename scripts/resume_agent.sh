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
for f in
.agent/START_HERE.md
.agent/00_state/PROJECT_STATE.md
.agent/00_state/SESSION_STATE.md
.agent/00_state/NEXT_ACTION.md
.agent/00_state/CLAIM_LEDGER.md
.agent/00_state/EXPERIMENT_LEDGER.md
.agent/00_state/TODO.md
do
if [ -f "$f" ]; then
echo "OK $f"
else
echo "MISS $f"
fi
done
echo

echo "## Latest Plan"
latest_plan=$(ls -t .agent/10_plans/*.md 2>/dev/null | head -1 || true)
if [ -n "${latest_plan:-}" ]; then
echo "$latest_plan"
echo
sed -n '1,80p' "$latest_plan"
else
echo "No plan found under .agent/10_plans/"
fi
echo

echo "## Latest Execution Report"
latest_exec=$(ls -t .agent/20_exec/*.md 2>/dev/null | head -1 || true)
if [ -n "${latest_exec:-}" ]; then
echo "$latest_exec"
echo
sed -n '1,100p' "$latest_exec"
else
echo "No execution report found under .agent/20_exec/"
fi
echo

echo "## Latest Verification Report"
latest_verify=$(ls -t .agent/30_verify/verify.md .agent/30_verify/VERIFY.md 2>/dev/null | head -1 || true)
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
echo " 1. Run: ./scripts/make_verify_packet.sh"
echo " 2. Ask Antigravity/Gemini/Codex to read prompts/verify_packet_prompt.md"
echo " 3. Commit only after PASS or PASS_WITH_MINOR_FIXES"
else
if [ -n "${latest_plan:-}" ]; then
echo "No uncommitted changes detected."
echo
echo "Recommended:"
echo " claude"
echo " /create-plan "create the next concrete plan based on current PROJECT_STATE, SESSION_STATE, and TODO""
else
echo "No plan found yet."
echo
echo "Recommended:"
echo " claude"
echo " /bootstrap-existing-project "scan this project, update state files, and recommend the first plan""
fi
fi

echo
echo "If you are lost, paste this into Antigravity or Claude CLI:"
echo
echo "Read prompts/resume_agent_prompt.md and resume this project. Update SESSION_STATE.md and NEXT_ACTION.md only."
echo
