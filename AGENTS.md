# Workspace working agreements

## Delegation

- The OpenAI model active in the parent thread is the orchestrator. It owns scope, planning, task assignment, integration, final review, verification, and communication with the user.
- Use T3 delegated child tasks through the `opencode` provider with these worker models:
  - `opencode/fledge-alpha-free` for implementation tasks.
  - `openrouter/thinkingmachines/inkling:free` for research and independent analysis.
  - `opencode/ling-3.1-flash-free` for testing and independent review.
- Give workers bounded, non-overlapping tasks with clear deliverables. The orchestrator reviews their work, reconciles changes, and verifies the integrated result. Do not use ordinary top-level threads as a substitute for delegated child tasks.
- If a listed worker is unavailable, report that and continue independent work where possible; do not silently substitute a different model.

## Commits

- Commit completed, verified work in small logical milestones as it progresses; do not leave all commits until the whole phase is finished.
- Write concise, human-sounding commit messages. Do not use conventional-commit prefixes such as `fix:`, `chore:`, or `feat:`.
