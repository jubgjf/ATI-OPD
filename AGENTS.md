# Repository Guidelines

## Project Structure & Module Organization

`verl/` contains the main Python package. Key areas include `verl/trainer/` for training orchestration and Hydra configs, `verl/workers/` for rollout/training workers, `verl/experimental/` for newer agent and distillation paths, and `verl/utils/` for shared helpers. Tests live under `tests/`, mirroring package namespaces; special suites use `tests/special_*`. User-facing examples and launch scripts are under `examples/` and `run/`. Documentation sources are in `docs/`, while container recipes live in `docker/`.

## Build, Test, and Development Commands

Install locally for Python-only development:

```bash
pip install -e .[test,vllm]
# or
pip install -e .[test,sglang]
```

Use focused checks during iteration:

```bash
python -m pytest tests/path/to/test_file_on_cpu.py -q
python -m ruff check verl tests
python -m ruff format verl tests
pre-commit run --all-files
```

Build docs from `docs/` with `make clean && make html`. Full distributed training usually requires GPU cluster resources; prefer module-level tests locally.

## Coding Style & Naming Conventions

Python uses 4-space indentation and a 120-column Ruff line length. Imports are sorted by Ruff/isort with `verl` as first-party. Prefer explicit, typed dataclasses/config objects and clear `ValueError`/`RuntimeError` messages at module boundaries. Test files intended for CPU CI should end with `_on_cpu.py`. Use the canonical project spelling `verl`; pre-commit also enforces `SGLang`/`sglang` naming.

## Testing Guidelines

Use `pytest`. Place tests near the matching namespace, for example `tests/trainer/` for `verl/trainer` behavior. CPU-safe tests should use the `test_*_on_cpu.py` suffix; GPU, distributed, e2e, and standalone cases belong in the relevant `tests/special_*` area or CI workflow. Add targeted tests for new config validation, tensor shape contracts, and backend integration assumptions.

## Commit & Pull Request Guidelines

Recent history uses scoped subjects such as `[fsdp] fix: ...`, `[data, rollout] feat: ...`, and `[doc] refactor: ...`. Keep commits focused and imperative. PRs should describe behavior changes, list tests run, link issues when relevant, and update docs/config examples for user-facing changes. For CI-impacting tests, update the matching workflow path filters.

## Agent-Specific Instructions

Before editing, inspect local patterns and avoid unrelated refactors. Do not revert user changes in a dirty worktree. Prefer `rg`, focused `pytest`, and `ruff` checks; do not launch full training unless explicitly requested and the environment supports it.
