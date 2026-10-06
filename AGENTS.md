# Workspace working agreements

## Delegation

- The OpenAI model active in the parent thread is the orchestrator. It owns scope, planning, task assignment, integration, final review, verification, and communication with the user.
- Form the worker team with T3 delegated child tasks through the `opencode` provider. Check `orchestrator_capabilities` for current availability before delegation; use these exact model IDs and role order:
  - Implementation: `opencode/space-bunny-free` → `opencode/fledge-alpha-free` → `opencode/muse-spark-1.3-contributor-free`.
  - Research and independent analysis: `openrouter/thinkingmachines/inkling:free` → `opencode/muse-spark-1.3-contributor-free` → `opencode/space-bunny-free`.
  - Testing and independent review: `opencode/ling-3.1-flash-free` → `openrouter/thinkingmachines/inkling:free` → `opencode/muse-spark-1.3-contributor-free`.
- Use the first available model in that role's sequence. If delegation fails because the model/provider is unavailable or returns an unusable result, record the failure and try the next listed model. Do not switch away from a still-running worker just because it is taking time, and do not start overlapping replacement work while the original task may still be active. If every listed option fails, continue as orchestrator where possible and report the limitation.
- Give workers bounded, non-overlapping tasks with clear deliverables. Include the user's task constraints (for example, prohibitions on Python, tests, or simulations) in each delegated prompt. The orchestrator reviews worker results, reconciles changes, and verifies the integrated result. Use delegated child tasks, not ordinary top-level threads, for subagent work.

## Commits

- Commit completed, verified work in small logical milestones as it progresses; do not leave all commits until the whole phase is finished.
- Write concise, human-sounding commit messages. Do not use conventional-commit prefixes such as `fix:`, `chore:`, or `feat:`.
