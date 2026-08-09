# Deprecated CoT Utility

The former standalone utility has been integrated into the main codebase. New
code should use `ldm_tts.data` and `scripts/augment_ldm_data.py`; this directory
is retained only so old paths fail gracefully or continue to delegate.

From the repository root:

```bash
export LLM_BASE_URL=https://your-openai-compatible-endpoint/v1
export LLM_API_KEY=...
export LLM_MODEL_NAME=DeepSeek-V4-Flash

python scripts/augment_ldm_data.py \
  --input data/ldm_ir.jsonl \
  --output data/ldm_ir_justified.jsonl \
  --sft-output data/ldm_sft_justified.jsonl
```

See `DATA_COLLECTION.md` for input formats, checkpoint behavior, schema-aware
skip rules, and the Python interface. Credentials must only be provided through
environment variables.
