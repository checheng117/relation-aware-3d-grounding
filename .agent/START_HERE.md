# START HERE

If you reopen this project and do not remember what to do, run:

    ./scripts/resume_agent.sh

Then follow the Next Recommended Action section.

## Roles

- Antigravity Ultra: Planner / Writer / Senior Reviewer
- Claude CLI: Executor / Repo Operator
- Codex CLI: Strict Verifier for important diffs only
- Gemini / Antigravity low-cost models: Fast Verifier

## Standard Workflow

1. Resume project state:
   ./scripts/resume_agent.sh

2. Create or choose a plan:
   .agent/10_plans/*.md

3. Execute with Claude CLI:
   /execute-plan .agent/10_plans/<plan>.md

4. Verify:
   ./scripts/make_verify_packet.sh

5. Commit only after review.

## If You Are Lost

Ask Antigravity or Claude CLI:

Read prompts/resume_agent_prompt.md and resume this project.
Update SESSION_STATE.md and NEXT_ACTION.md only.
Do not modify source code or paper text.
