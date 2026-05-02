# Resume Agent Prompt

You are resuming a half-finished research-agent project.

Read:
- .agent/START_HERE.md
- .agent/00_state/PROJECT_STATE.md
- .agent/00_state/SESSION_STATE.md
- .agent/00_state/NEXT_ACTION.md
- .agent/00_state/CLAIM_LEDGER.md
- .agent/00_state/EXPERIMENT_LEDGER.md
- .agent/00_state/TODO.md
- latest .agent/10_plans/*.md
- latest .agent/20_exec/*.md
- latest .agent/30_verify/*.md if present

Your task:
1. Explain where the project currently stands.
2. Identify whether there is an uncommitted diff.
3. Identify the latest plan and whether it was executed.
4. Identify whether verification is needed.
5. Update SESSION_STATE.md and NEXT_ACTION.md.
6. Recommend exactly one next action.

Rules:
- Do not modify source code.
- Do not modify paper text.
- Do not invent results.
- If unclear, write UNKNOWN.
