# Protein Inverse Folding

This task registers the MLS-Bench
`ai4bio-protein-inverse-folding` architecture-design benchmark behind the shared
LDM-TTS runner. The model proposes Python for `StructureEncoder` and
`InverseFoldingModel`; the task validates the proposal, inserts it into the fixed
benchmark scaffold, evaluates recovery and perplexity, and retains the highest
observed aggregate score.

Source: https://github.com/Imbernoulli/MLS-Bench/tree/main/tasks/ai4bio-protein-inverse-folding

## Contract

Candidates receive backbone coordinates `X` shaped `(B, L, 4, 3)` for N, CA,
C, and O atoms plus a residue mask shaped `(B, L)`. `StructureEncoder` returns
`(B, L, hidden_dim)`. `InverseFoldingModel` returns normalized log
probabilities shaped `(B, L, 20)`.

The model response is JSON with `reasoning`, `summary`, and complete `code`.
The code must define both required classes and one literal `CONFIG_OVERRIDES`
dict. Only `learning_rate`, `dropout`, `num_encoder_layers`, and `batch_size`
may be overridden. The data loader, splits, training loop, loss, metrics, and
test harness remain fixed. The upstream 1.05x-largest-baseline rule is enforced
at 4,491,989 trainable parameters before benchmark training begins.

The aggregate objective follows the public score specification. Recovery and
perplexity are direction-normalized between the worst baseline and theoretical
bound 1, with the best baseline calibrated to score 0.5. Each benchmark score
is the equally weighted mean of those two terms; the task score is the
geometric mean over CATH4.2, CATH4.3, and TS50. The pinned baseline rows are in
`resources/baseline_leaderboard.csv`.

## Mock Run

The mock path uses only the lightweight shared runtime and NumPy dependency. It
parses and validates real candidate-shaped code, evaluates deterministic AST
features, and writes the same run manifest as a real search. It does not import
PyTorch, contact a model endpoint, download data, or use a GPU.

```bash
uv sync --locked --project tasks/protein_inverse_folding --group dev
uv run --locked --project tasks/protein_inverse_folding \
  python scripts/run_ldm_tts.py config/protein_inverse_folding/mock.yaml
```

## Data Collection

Validated model actions are collectable as canonical `ldm-2.0` complete-design
IR. Collection is opt-in through `LDM_DATA_COLLECTION_ENABLED=1` or
`LDM_DATA_COLLECTION_DIR`. The default location is
`runs/<run>/ldm_data/`. Rejected JSON, invalid Python, and unvalidated fallbacks
are never collected. Run identity and evaluator outcomes are stored only under
the collection-only metadata field and are excluded from rendered SFT prompts.

## GPU Contract Smoke

The GPU smoke instantiates the assembled encoder, performs a forward pass,
checks `(B, L, 20)` output and log-probability normalization, and runs backward
on every requested device. It needs PyTorch with CUDA but no PInvBench data.

```bash
uv run --locked --group real --project tasks/protein_inverse_folding \
  python scripts/check_task_dependencies.py \
  config/protein_inverse_folding/real_gpu_smoke.yaml

uv run --locked --group real --project tasks/protein_inverse_folding \
  python scripts/run_ldm_tts.py \
  config/protein_inverse_folding/real_gpu_smoke.yaml
```

## Full Benchmark Requirements

The full evaluator intentionally does not vendor MLS-Bench, PInvBench, or the
CATH/TS datasets. Prepare:

- an OpenAI-compatible endpoint through `TTS_LLM_URL`, `TTS_LLM_MODEL`, and
  `TTS_LLM_API_KEY` or `OPENAI_API_KEY`;
- the upstream `edits/custom_template.py` fixed scaffold, or the
  `ProteinInvBench/custom_invfold.py` file created from it by MLS-Bench;
- a data root containing `cath4.2/`, `cath4.3/`, and `ts/` in the layout read by
  PInvBench;
- a Python environment where `torch`, CUDA, NumPy, and `PInvBench` are
  importable.

Generated model code executes during evaluation. Use a trusted model endpoint
and an isolated compute environment.

## Staged First Real Run

First, check the endpoint and code contract without training or datasets:

```bash
python scripts/check_task_dependencies.py \
  config/protein_inverse_folding/real.yaml \
  --set args.iterations=0 \
  --set args.skip-eval=true \
  --no-optional

python scripts/run_ldm_tts.py config/protein_inverse_folding/real.yaml \
  --set args.iterations=0 \
  --set args.skip-eval=true \
  --set args.run-name=protein_inverse_folding_contract
```

Next, run the GPU contract config. Then check all full dependencies with real
paths:

```bash
python scripts/check_task_dependencies.py \
  config/protein_inverse_folding/real.yaml \
  --set args.scaffold-path=/path/to/custom_invfold.py \
  --set args.data-root=/path/to/data
```

Finally, perform one evaluated proposal before increasing the search budget:

```bash
python scripts/run_ldm_tts.py config/protein_inverse_folding/real.yaml \
  --set args.iterations=1 \
  --set args.breadth=1 \
  --set args.scaffold-path=/path/to/custom_invfold.py \
  --set args.data-root=/path/to/data \
  --set args.run-name=protein_inverse_folding_tiny
```

The public benchmark defaults can take up to 3 hours for each CATH run and 6.5
hours for TS50. The real config assigns the three benchmarks to separate GPUs
and evaluates them in parallel.
