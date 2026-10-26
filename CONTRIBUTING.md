# How to Contribute

Thanks for your interest in contributing to `causal-ts`! Here are a few
general guidelines on contributing and reporting bugs that we ask you to review.
Following these guidelines helps to communicate that you respect the time of the
contributors managing and developing this open source project. In return, they
should reciprocate that respect in addressing your issue, assessing changes, and
helping you finalize your pull requests. In that spirit of mutual respect, we
endeavor to review incoming issues and pull requests within 10 days, and will
close any lingering issues or pull requests after 60 days of inactivity.

Please note that all of your interactions in the project are subject to our
[Code of Conduct](/CODE_OF_CONDUCT.md). This includes creation of issues or pull
requests, commenting on issues or pull requests, and extends to all interactions
in any real-time space e.g., Slack, Discord, etc.

## Reporting Issues

Before reporting a new issue, please ensure that the issue was not already
reported or fixed by searching through our [issues
list](https://github.com/bloomberg/causal-ts/issues).

When creating a new issue, please be sure to include a **title and clear
description**, as much relevant information as possible, and, if possible, a
test case.

**If you discover a security bug, please do not report it through GitHub.
Instead, please see security procedures in [SECURITY.md](/SECURITY.md).**

## Sending Pull Requests

Before sending a new pull request, take a look at existing pull requests and
issues to see if the proposed change or fix has been discussed in the past, or
if the change was already implemented but not yet released.

We expect new pull requests to include tests for any affected behavior, and, as
we follow semantic versioning, we may reserve breaking changes until the next
major version release.

## Notebooks (Jupytext)

Every example notebook `examples/<name>.ipynb` is paired with a plain-text
mirror `examples/scripts/<name>.py` using
[Jupytext](https://github.com/jupytext/jupytext), so code changes show up as
reviewable `.py` diffs instead of notebook JSON. Pairing is configured once
in `[tool.jupytext]` in `pyproject.toml`; `tutorial.ipynb` and notebooks
outside `examples/` are not paired.

- **Setup (recommended)**: `pip install -e ".[dev]" pre-commit` then
  `pre-commit install`. The hook runs `jupytext --sync` on every commit that
  touches a notebook or mirror; if it updates the other file, stage it and
  commit again.
- **Without the hook**: after editing either side, run
  `jupytext --sync examples/<name>.ipynb` and commit both files.
- **New notebook**: add `examples/<name>.ipynb` and sync it; the mirror is
  created in `examples/scripts/`. Deleting or renaming a notebook means doing
  the same to its mirror.
- **Outputs and widget state** live only in the `.ipynb`. Read the Docs
  renders the committed notebooks (`nb_execution_mode = "off"` in
  `docs/conf.py`) and the mirror never carries outputs, so output churn stays
  out of review diffs. Syncing preserves outputs.
- **Lint**: do not run `black` or `isort` on the mirrors (both are configured
  to skip them): reordering imports across `# %%` cell markers corrupts the
  notebook on the next sync. flake8 checks mirrors only for syntax errors and
  undefined names, matching the notebooks' lint exemption in `lint.yml`.
- **CI**: the `python-lint` job re-syncs every example notebook and fails if
  a pair has drifted, a mirror is missing, or a mirror has no notebook.
  Jupytext is pinned (dev extra, CI, and `.pre-commit-config.yaml`) because
  the mirror header records the generating version; bump all three together
  and re-sync.

## Other Ways to Contribute

We welcome anyone that wants to contribute to `causal-ts` to triage and
reply to open issues to help troubleshoot and fix existing bugs. Here is what
you can do:

- Help ensure that existing issues follows the recommendations from the
  _[Reporting Issues](#reporting-issues)_ section, providing feedback to the
  issue's author on what might be missing.
- Review and update the existing content of our
  [Wiki](https://github.com/bloomberg/causal-ts/wiki) with up-to-date
  instructions and code samples.
- Review existing pull requests, and testing patches against real existing
  applications that use `causal-ts`.
- Write a test, or add a missing test case to an existing test.

Thanks again for your interest on contributing to `causal-ts`!

:heart:
