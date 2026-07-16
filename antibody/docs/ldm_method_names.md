# LDM/LLM Method Names

This note keeps method names stable while preserving the historical code paths.

| Code path | Old label | Initialization | Point selection | Stable method name |
| --- | --- | --- | --- | --- |
| `bo/ldm` | LDM | random/LHS | LLM sequentially updates one search-function DSL, then acquisition argmax | `LDM_fn_seq_argmax` |
| `bo/ldm_reservoir` | LDM+Acq softmax | random | LLM samples parallel search-function strategies, then acquisition softmax | `LDM_fn_par_softmax` |
| `bo/ldm_reservoir` | LDM+Acq argmax | random | LLM samples parallel search-function strategies, then acquisition argmax | `LDM_fn_par_argmax` |
| `bo/ldm_light/ldm_acq.py` | LDM prior+Acq | LLM generated | LLM generates one search-function DSL, then acquisition argmax | `LDM_fn_one_argmax` |
| `bo/ldm/llm/LLM_baseline.py` | Pure LLM | LLM rerank | LLM reranks/selects from a candidate pool | `LLM_rerank` |
| `bo/llm_direct` | Direct LLM | LLM generated | LLM directly generates next antibody JSON list | `LLM_gen` |
| `bo/llm_direct` | Direct LLM + acquisition | LLM generated | LLM samples `m` antibodies, acquisition softmax selects | `LDM_gen_softmax` |
| `bo/llm_direct` | Direct LLM + acquisition | LLM generated | LLM samples `m` antibodies, acquisition argmax selects | `LDM_gen_argmax` |

The new direct-generation methods are run through:

```bash
python scripts/run_llm_direct_absolut.py --method LLM_gen ...
python scripts/run_llm_direct_absolut.py --method LDM_gen_softmax ...
python scripts/run_llm_direct_absolut.py --method LDM_gen_argmax ...
```

All three methods write AntBO-compatible `results.csv` files.
