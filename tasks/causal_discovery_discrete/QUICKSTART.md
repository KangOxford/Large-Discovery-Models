# Clean-Room Quick Start

1. Check out MLS-Bench at `cfd57a7e0139c72753e32e31bca593719b098717`.
2. Install this task environment with `uv sync --project tasks/causal_discovery_discrete`.
3. Run the mock config and inspect its durable artifacts.
4. Set `upstream-root` in a private real config and run dependency pre-flight.
5. Dry-run the selected contract profile before starting real evaluation.

```bash
python3 scripts/validate_tasks.py --task causal_discovery_discrete
uv run --locked --project tasks/causal_discovery_discrete python3 -m pytest tasks/causal_discovery_discrete/tests
uv run --locked --project tasks/causal_discovery_discrete python3 scripts/check_task_dependencies.py config/causal_discovery_discrete/mock.yaml --no-optional
uv run --locked --project tasks/causal_discovery_discrete python3 scripts/run_ldm_tts.py config/causal_discovery_discrete/mock.yaml --dry-run
```
