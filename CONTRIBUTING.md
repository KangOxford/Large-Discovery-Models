# Contributing

Thank you for contributing to Large Discovery Models. Keep changes focused,
reproducible, and explicit about which claims have been verified.

## Development environments

The shared runner and each scientific task use separate locked environments.
Set up the root environment with:

```bash
uv sync --locked --group dev
```

Task-specific commands and coverage thresholds are documented in
[testing guide](docs/testing.md). Run the lanes affected by your change. Do not merge
the task dependency stacks into the root package.

## Task contributions

New tasks must include a `task.json` manifest, a shallow `ldm_task` adapter,
tests, mock configuration, dependency checks for real runs, and documentation.
Use `experiment.json` for scientific provenance, metric roles, evaluator
settings, budgets, and qualification state. A runnable mock is not sufficient
to mark a task `qualified`; qualification must be supported by the documented
real-evaluator evidence.

Run `uv run --locked python scripts/validate_tasks.py` before submitting a
task change. See [`tasks/README.md`](tasks/README.md) for the complete contract.

## Pull requests

- Explain the behavior changed and the verification performed.
- Add focused tests for new behavior and regressions.
- Keep generated runs, caches, credentials, local paths, and large unreviewed
  artifacts out of Git.
- Keep the G12D joblib artifact external; the repository tracks only its
  metadata and expected checksum.
- Document the provenance, checksum, format, and redistribution terms of any
  committed data or model artifact.
- Never load pickle-compatible artifacts from an untrusted source.

By participating, you agree to follow the repository's
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
