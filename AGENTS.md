# Instructions for coding agents

These apply to the whole repository. Human contributors should also read
[CONTRIBUTING.md](CONTRIBUTING.md); the checklist below applies to them too.

## Before you edit

- Read the code you are changing, its tests, and any docs or example notebooks
  that call it. A behavior change often breaks a notebook in `examples/`.
- Keep the change focused: no unrelated refactoring or reformatting.

## While you edit

- Add or update tests in `tests/` for any changed behavior.
- Code must run on Python 3.10 (CI tests 3.10 to 3.14), so avoid newer syntax
  and APIs such as `except*`, `typing.Self`, `tomllib`, `type` aliases or
  PEP 695 generics.
- New Python files under `causalts/`, `tests/`, `docs/` or `experiments/` start
  with:

  ```python
  # Copyright <current year> Bloomberg Finance L.P.
  # SPDX-License-Identifier: GPL-3.0-or-later
  ```

- Add a line for user-visible changes under `## [Unreleased]` in
  `CHANGELOG.md` (`### Added`, `### Changed` or `### Fixed`).
- If `examples/scripts/` exists, each notebook `examples/<name>.ipynb` is
  paired with `examples/scripts/<name>.py` by Jupytext. After editing either
  side, run `jupytext --sync examples/<name>.ipynb` and commit both files. Do
  not run black or isort on `examples/scripts/`. If `.pre-commit-config.yaml`
  exists, `pre-commit install` sets up a hook that syncs for you.
- Re-run only the notebooks you changed, top to bottom. Graph plots default to
  the Graphviz `neato` layout and fall back to a circular layout without it;
  the warning is hidden in most notebooks. First install Graphviz and
  pygraphviz (`brew install graphviz` or `apt install graphviz libgraphviz-dev`,
  then `pip install pygraphviz`) and check with `dot -V` and
  `python -c "import pygraphviz"`. If you can't, don't commit the re-run
  outputs of graph cells, and say so.

## Before you hand back

Run these and fix what fails; CI runs the same lint checks.

```bash
pip install -e ".[dev]" "black==24.*" Flake8-pyproject codespell
python -m pytest tests/test_<module>.py   # tests for what you changed
black --check .                           # CI pins black 24; newer versions format differently
isort --check .
flake8 .
codespell --config .github/linters/.codespellrc <changed files>
jupytext --sync examples/<name>.ipynb  # each notebook you changed, if paired
```

If `examples/scripts/` exists, every notebook you changed must then appear in
`git status` together with its `examples/scripts/` twin.

Run the full suite (`python -m pytest tests/`) when you can; it is slow.
Never say a check passed unless you ran it, and say which checks you skipped.

## Pull request

- Fill in `.github/pull_request_template.md`, link the issue it fixes, and
  say in the description if AI tools wrote a significant part of the change.
- Do not take GitHub actions (push, open PRs, comment, star) the user did not
  ask for.
