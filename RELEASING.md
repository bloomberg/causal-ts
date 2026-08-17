# Releasing `causalts` to PyPI

Publishing is automated by `.github/workflows/release.yml` using PyPI
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC). No API
token or secret is stored in the repository.

## One-time setup

### 1. PyPI Trusted Publisher

On <https://pypi.org>, sign in and add a **pending** publisher
(Account settings → Publishing → Add a pending publisher) with:

| Field             | Value         |
| ----------------- | ------------- |
| PyPI project name | `causalts`    |
| Owner             | `bloomberg`   |
| Repository name   | `causal-ts`   |
| Workflow filename | `release.yml` |
| Environment name  | `pypi`        |

`Owner` is the **GitHub org that owns the repository**
(`github.com/bloomberg/...`),
not your PyPI username. The GitHub OIDC token asserts
`repository_owner = bloomberg`, and PyPI matches it against this value.

### 2. GitHub `pypi` environment

In the repository, go to **Settings → Environments → New environment**, name it
`pypi`, and add protections:

- **Required reviewers** — a maintainer must approve before the `publish` job
  runs (a human gate on every PyPI push).
- **Deployment branches and tags** — restrict to protected tags matching `v*`.

## Cutting a release

1. Bump `project.version` in `pyproject.toml` (e.g. `0.25.1` → `0.25.2`) and
   merge it to `main`. Confirm the `test` matrix is green. In the same PR, bump
   the other version-bearing files — they are **not** derived from
   `pyproject.toml` and go stale silently:
   - `CHANGELOG.md` — a new section plus the compare links at the bottom
   - `docs/_static/switcher.json` — only the stable entry's `name` (the label
     in the dropdown). Its `version` keys are Read the Docs slugs (`stable`,
     `latest`), not release numbers, so no new entry is needed per release
   - `docs/conf.py` — the `announcement` ribbon, if the release is worth
     advertising
   - `.claude-plugin/plugin.json` — the packaged skill's own version
2. Create and publish a **GitHub Release** whose tag is the version prefixed
   with `v` (e.g. `v0.25.2`). The tag must exactly match `pyproject.toml`;
   `release.yml` fails the build otherwise.
3. The workflow runs tests → builds sdist/wheel → smoke-tests the wheel on
   Python 3.10–3.14 → waits for `pypi` environment approval → publishes.

## Dry run (no publish)

Trigger the `release` workflow manually from the **Actions** tab
(`workflow_dispatch`). It runs tests, builds, and smoke-tests the wheel but
**skips publishing** (the `publish` job only runs on a `release` event).

## Notes

- Python 3.10–3.13 run the full optional-extra profile
  (`.[dev,dowhy,tigramite]`); Python 3.14 runs the core profile (`.[dev]`)
  because DoWhy does not yet advertise 3.14 support. The wheel is still
  smoke-tested on 3.14.
- The existing `v0.25.0` GitHub Release predates this workflow and will not
  trigger it retroactively; the first automated publish is `v0.25.1`.
