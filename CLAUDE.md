# FillOrBust

A fresh model of the dice game Fill or Bust (also called Muss Buss). The root is
a blank `fillorbust` uv package (`src/fillorbust/`); set it up with `uv sync` and
test with `uv run pytest`.

## Python conventions

- uv manages the environment. Change dependencies only with `uv add` /
  `uv remove` (`--dev` for tooling); never hand-edit dependency lists in
  `pyproject.toml`.
- Run tools through `uv run`. Before calling work done, run and fix:
  `uv run ruff format`, `uv run ruff check`, `uv run ty check`, `uv run pytest`.
- Fully type annotate all code, tests included: every parameter and return
  type, plus attributes and empty containers whose type can't be inferred.
  Prefer precise types over `Any`; describe untyped third-party objects with a
  `Protocol`. ruff's `ANN` rules enforce this.
- New behavior gets tests in `tests/`.

## Code style

- No emojis in code, comments, commit messages, or docs.
- Keep comments concise and describe the code as it is now, never past
  iterations or why it changed. Git history covers that.

## Git

- Work in atomic commits: one logical change per commit, each passing the
  checks above. Split unrelated changes into separate commits.
- pre-commit runs ruff, ty and `uv lock --check` on every commit, and pytest on
  push.
- The default branch is `main`.

## Project notes

- `archive/` holds the previous attempts, kept for reference only. ruff, ty,
  pytest and pre-commit all exclude it. Do not edit, lint, format or modernize
  archived code.
- Archive layout is one folder per era, `archive/<year>-<name>/`:
  `2017-fillorbust-cv`, `2018-mussbuss-scratch`, `2019-mussbot`,
  `2021-gamelearning`, `2021-simpledice`, `2025-dice`. `archive/README.md`
  records each one's origin, dates and Python version.
- Each era was merged with its git history; `git log -- archive/<era>` shows it.
  The 2017 era's history is on the commits before it was moved into `archive/`.
- The tag `archive/gamelearning-simplify-fob` preserves GameLearning's unmerged
  `simplify-fob` branch (dice paths only).
- `archive/2025-dice/pyproject.toml` is a separate Poetry project, not a uv
  workspace member. Archived projects keep their original packaging files.
- The new `fillorbust` package must not import from `archive/`. Port ideas by
  rewriting them, not by importing or copying modules.
