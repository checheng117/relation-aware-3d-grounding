
NEXT_ACTION
Next Recommended Action

Run bootstrap/resume scan if this file has not been updated recently.

Command
./scripts/resume_agent.sh
If You Are Using Claude CLI
/bootstrap-existing-project "resume this half-finished research project, update SESSION_STATE and NEXT_ACTION, and recommend the next concrete plan. Do not modify source code or paper text."
If You Are Using Antigravity

Ask it to read:

.agent/START_HERE.md
.agent/00_state/PROJECT_STATE.md
.agent/00_state/SESSION_STATE.md
.agent/00_state/NEXT_ACTION.md
latest .agent/20_exec/*.md
latest .agent/30_verify/*.md

Then ask it to recommend the next plan.
