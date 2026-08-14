# Versioned resources

`upstream_contract.json` records the immutable primary-source contract used by
the dependency checker and evaluator. `seed_quantizer.py` is byte-for-byte the
official `AdaptiveKVQuantizer` class from editable lines 41-172 at the pinned
commit; deterministic seed evaluations do not enter the fine-tuning collection
stream.

Runtime candidates and evaluator copies belong under ignored `runs/`
directories. The MLS-Bench checkout, `transformers-kv-lab` package, model, and
datasets remain external and must satisfy `upstream_contract.json`.
