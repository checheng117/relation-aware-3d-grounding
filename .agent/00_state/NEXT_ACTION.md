# NEXT_ACTION

## Next Recommended Action

Run:

    ./scripts/resume_agent.sh

Then inspect whether there are uncommitted changes.

## If There Are Uncommitted Changes

1. Inspect:
   git status --short

2. If tracked files were accidentally deleted, restore them:
   git restore -- $(git diff --name-only --diff-filter=D)

3. Generate verifier packet:
   ./scripts/make_verify_packet.sh

4. Ask Antigravity, Gemini, or Codex to verify the packet.

## If There Is No Plan

Run in Claude CLI:

    /bootstrap-existing-project "scan this project, update state files, and recommend the first plan"

## If There Is A Plan

Run in Claude CLI:

    /execute-plan .agent/10_plans/<plan>.md

## Do Not

- Do not run git add . blindly.
- Do not commit accidental deletions.
- Do not let agents invent experimental numbers.
- Do not launch long GPU jobs unless a plan explicitly allows it.
