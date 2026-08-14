# Quickstart

1. Create the task environment from the checked-in lock and run the required
   registration, task-test, dependency, dry-run, and mock gates shown in
   `README.md`.
2. Run `preflight.yaml` with an evaluator Python that imports PyTorch. This
   verifies CPU tensor shape, dtype, device, finite values, bit accounting, and
   retained tensor state without downloading the model or datasets.
3. Check out MLS-Bench at
   `cfd57a7e0139c72753e32e31bca593719b098717` and prepare its
   `transformers-kv-lab` environment.
4. Run `official_seed.yaml` with `upstream-root`, `package-dir`, and
   `evaluator-python` set. This is a five-GPU, five-workload official-budget
   evaluation.
5. Configure the OpenAI-compatible proposal endpoint only through
   `LDM_LLM_URL`, `LDM_LLM_MODEL`, and `LDM_LLM_API_KEY`.
6. Run `tiny_real.yaml`. Confirm one generated reservoir was scored, exactly one
   selected candidate entered the evaluator, and the run remained labeled
   non-official.
7. Run `extended_tiny_real_20.yaml` only for extended operational coverage.
   Confirm 20 endpoint requests produced 20 four-candidate reservoirs, 20
   GP-UCB selections, and 20 one-example HotpotQA jobs. This profile remains
   non-official and is not benchmark-comparable.
8. Run `official_campaign.yaml` only when five GPUs and the full official data
   are available. Monitor durable status and budget files; resume an interrupted
   directory with `--set args.resume-from=/absolute/run/path`.
9. Inspect `search_manifest.json`, `selection_record.json`,
   `evaluation_manifest.json`, workload logs, `summary.json`, and the immutable
   contract snapshot before making any comparison claim.

Do not change `qualification` to `qualified` based on registration, mock,
tensor preflight, a one-workload run, or an unselected standalone candidate.
