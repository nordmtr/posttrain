# AGENTS.md

You are a code-writing agent operating in this repo. Follow these rules strictly.

## 0) Prime directive

* Produce small, correct, reviewable changes.
* Never “assume it works” — verify it.
* Prefer boring, explicit code. No overengineering, no gratuitous abstractions.
* **Don’t touch unrelated files.** Keep diffs tight.

## 1) Environment + dependencies (uv only)

* This repo uses **uv** for venv + dependency management.
* Do NOT use `pip install` directly.
* Add/remove dependencies via uv (don’t hand-edit dependency lists):

  * Runtime: `uv add <pkg>` / `uv remove <pkg>`
  * Dev: `uv add --dev <pkg>` / `uv remove --dev <pkg>`
* Install/sync: `uv sync`
* Run commands: `uv run <cmd>`

If you modify dependencies, ensure:

* `pyproject.toml` updated
* `uv.lock` updated (via uv commands)
* checks + tests still pass

If you must edit `pyproject.toml` manually (rare), you **must** reconcile with uv afterward (lock/sync) so `uv.lock` matches.

## 2) Read before you write

Before starting work:

* Skim `docs/KNOWLEDGE.md` for repo gotchas / constraints.
* Skim `docs/ARCHITECTURE.md` if you’re touching core flows, entrypoints, jobs, or integrations.

## 3) Closed feedback loop (mandatory)

For every meaningful code edit (logic, behavior, deps, refactors touching execution paths):

1. Implement the smallest viable change.
2. Run the smallest relevant verification:

   * Prefer a targeted test: `uv run pytest -q path/to/test.py::test_name`
   * Or a minimal run command that exercises the changed path.
3. If it fails, debug and repeat until green.
4. Only then proceed.

**Honesty rule:** Never claim you ran commands you didn’t run. If you can’t run something (missing services/creds/platform limits), say so and use the next-best verification (unit tests, mocks, static checks) and note the risk.

## 4) Tests are not optional

* Any behavior change must be covered by tests.
* Any bug fix must include a regression test that fails before the fix and passes after.
* Keep tests focused and fast. Prefer unit tests; mock external services.

If the repo has no tests yet:

* Add minimal pytest scaffolding and at least one test for the new behavior.

## 5) Error recovery (avoid the spiral of death)

If you hit a failing error and you’re iterating on the same failure:

* Track each attempt as: **hypothesis → change → verification result** (keep it short).
* **Three strikes rule:** after 3 failed attempts on the same error, stop.

  * Revert to the last known-green state (or clearly identify it).
  * Write a new plan (different approach).
  * If still blocked, ask for developer input with the minimal context needed (error, last command, what you tried).

## 6) Quality gates (levels)

We use three verification levels. Choose the smallest level that matches the risk.

### A) Loop gate (during iteration) — required

Run **one** of:

* A targeted test for the changed behavior, or
* A minimal run command that exercises the change

### B) Commit gate (before each commit) — required

Run:

1. **Targeted tests** for changed behavior (can be a small subset)
2. **Format:** `uv run ruff format`
3. **Lint:** `uv run ruff check`

   * If you intend auto-fixes, use: `uv run ruff check --fix`
4. **Type check:** `uv run pyrefly check`

Re-run targeted tests if formatting/lint/type checks changed code.

### C) Pre-push / PR gate (before opening PR or handing off) — required

Run the full suite:

* `uv run pytest`
* plus the Commit gate checks above (if not already done after the last code change)

**When to escalate to full `pytest` earlier (even before commit):**

* dependency changes
* refactors touching shared/core modules
* changes affecting many call sites
* anything that feels “maybe subtle”

## 7) Repo hygiene + safety

* Don’t log, print, paste, or commit secrets (API keys, tokens, credentials).
* Don’t add real `.env` files to git. Prefer `.env.example` + docs.
* Don’t reformat, rename, or “cleanup” the repo unless it’s necessary for the task.
* Preserve existing conventions and patterns; search the repo before inventing a new style.

## 8) How to work (plan → implement → verify → commit)

For non-trivial tasks:

1. Write a brief plan (bullets).
2. Implement in small steps.
3. After each step: run a Loop gate verification (Section 6A).
4. When a step is complete and stable: run Commit gate (Section 6B) and commit.
5. Before handoff / PR: run Pre-push gate (Section 6C).

## 9) Documentation duties

We maintain three docs. Update them when relevant:

### AGENTS.md (this file)

* Only the developer should change policy here.
* You may propose edits, but don’t modify it unless asked.

### docs/KNOWLEDGE.md (agent → agent memory)

Update when:

* you discover a recurring gotcha / constraint
* you make a non-obvious decision
* you learn repo-specific workflow details

Keep it short and actionable. Not a diary.

### docs/ARCHITECTURE.md (agent → developer map)

Update when:

* you add/remove modules
* you change a core flow
* you add new entrypoints / jobs / integrations

Every claim must include file pointers (paths + key symbols).

## 10) Git commits (required)

Policy:

* Each standalone edit that works and passes gates should be committed.
* Keep commits small but meaningful (avoid 20 micro-commits for one rename).
* If changes are unrelated, split into separate commits.

Commit message format:

* If the repo is new or already uses gitmoji, use: `:emoji: <type>: <short summary>`

  * Examples:

    * `:sparkles: feat: add user signup endpoint`
    * `:bug: fix: handle empty input in parser`
    * `:recycle: refactor: simplify config loading`
    * `:white_check_mark: test: add regression for auth middleware`
    * `:memo: docs: update architecture map`
* Otherwise, follow the repo’s existing commit convention.

Rules:

* Imperative voice, specific, no essays.

## 11) Finish output format

When you finish a task, include:

* What changed (bullets)
* Commands run + results (what “success” looked like)
* Risks / follow-ups
* Whether `docs/KNOWLEDGE.md` / `docs/ARCHITECTURE.md` were updated
