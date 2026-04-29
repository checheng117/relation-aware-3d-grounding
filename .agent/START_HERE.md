# START HERE

If you reopen this project and do not remember what to do, run:

```bash
./scripts/resume_agent.sh

Then follow the "Next Recommended Action" section.

Workflow Roles
Antigravity Ultra:
Planner
Writer
Senior Reviewer
Claude CLI:
Executor
Repo Operator
Codex CLI:
Strict Verifier for important diffs only
Gemini / Antigravity low-cost models:
Fast Verifier
Standard Workflow
Resume project state:
./scripts/resume_agent.sh
If no clear next step:
Ask Antigravity or Claude CLI to read:
.agent/00_state/PROJECT_STATE.md
.agent/00_state/SESSION_STATE.md
.agent/00_state/NEXT_ACTION.md
.agent/00_state/TODO.md
Create or choose a plan:
.agent/10_plans/*.md
Execute with Claude CLI:
/execute-plan .agent/10_plans/<plan>.md
Verify:
./scripts/make_verify_packet.sh
Commit only after review.
