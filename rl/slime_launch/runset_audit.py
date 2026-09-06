#!/usr/bin/env python3
"""Field-coverage scan of the _wt_handoff EXAMPLE artifacts. LIMITED SCOPE.

CORRECTED 2026-09-06. The previous version initialised every required field to None, never
searched for most of them, and then reported the untouched Nones as ABSENT -- while reading
only _wt_handoff example summary/result files, no run directories, logs, configs, caches or
source. It also mixed in non-small-molecule examples.

This version reports an explicit evidence state per field and CANNOT be read as a statement
about all existing data:
  NOT_CHECKED                  this script does not look for it
  NOT_RECORDED_IN_FIELDS_READ  searched in the files read here; not present there
  CONFIRMED_FROM_<source>      found, with the source named

The duplicate-count reconciliation formerly in this file is WITHDRAWN and moved to
plan/dup_reconcile.py, which reads the actual gp_history files. The claim that the counts
were "two stages of one pipeline" contradicted B1 progress.md 380-388 and is retracted.
"""
import json, glob, re, hashlib, subprocess
from pathlib import Path

L = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model")
T = L/"ldm_rl"
W = L/"_wt_handoff"
OUT = {"generated_utc": "2026-09-06", "note":
       "Fields recorded as null are ABSENT from the artifact, not unknown to the auditor."}

REQUIRED = ["run_id","model_size","init_from","train_seed","eval_seed","checkpoint",
            "scorer","code_hash","prior_manifest","smiles_raw","smiles_canonical",
            "succeeded_field","failure_types","budget_charge_per_call","upstream_dedupe",
            "stop_reason"]

def sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else None

# ---- 1. evaluation artifacts ---------------------------------------------------
arts = []
for f in sorted(glob.glob(str(W/"assets/examples/**/summary.json"), recursive=True)) + \
         sorted(glob.glob(str(W/"ready2run_examples/**/result.json"), recursive=True)):
    try: d = json.loads(Path(f).read_text())
    except Exception: continue
    if d.get("task") != "small_molecule": continue   # was: also accepted task=None
    # every required field starts as NOT_CHECKED, never as a bare None that later
    # reads as "absent"
    rec = {k: "NOT_CHECKED" for k in REQUIRED}
    rec["run_id"] = str(Path(f).parent.name)
    rec["artifact"] = str(Path(f).relative_to(L))
    rec["artifact_sha256_16"] = sha(f)
    # only what the file actually carries
    rec["stop_reason"] = d.get("stop_reason") or d.get("early_stop_reason")
    rec["succeeded_field"] = ("AGGREGATE_ONLY_summary.json:successful_evaluation_count -- not a per-call predicate; the per-call contract is engine_adapters.py:265-289 (vina AND activity non-None)"
                              if "successful_evaluation_count" in d
                              else "NOT_RECORDED_IN_FIELDS_READ")
    rec["successful_evaluation_count"] = d.get("successful_evaluation_count")
    rec["failed_evaluation_count"] = d.get("failed_evaluation_count")
    rec["observation_count"] = d.get("observation_count")
    rec["llm_call_count"] = d.get("llm_call_count")
    rec["drop_counts"] = d.get("drop_counts")
    rec["final_hypervolume"] = d.get("final_hypervolume")
    rec["method"] = d.get("method")
    rec["upstream_dedupe"] = ("CONFIRMED_FROM_summary.json:drop_counts.duplicate"
                              if isinstance(d.get("drop_counts"), dict)
                              and "duplicate" in d["drop_counts"]
                              else "NOT_RECORDED_IN_FIELDS_READ")
    # history: what per-record fields exist
    h = Path(f).parent/"history.json"
    if h.exists():
        try:
            hh = json.loads(h.read_text())
            if isinstance(hh, list) and hh and isinstance(hh[0], dict):
                rec["history_record_fields"] = sorted(hh[0].keys())
                rec["history_len"] = len(hh)
                sm = [r.get("smiles") for r in hh if isinstance(r, dict) and r.get("smiles")]
                rec["smiles_raw"] = ("CONFIRMED_FROM_history.json" if sm
                                     else "NOT_RECORDED_IN_FIELDS_READ")
                rec["history_unique_smiles_raw"] = len(set(sm)) if sm else None
        except Exception:
            pass
    arts.append(rec)
OUT["evaluation_artifacts"] = arts

# ---- 3. what the comparison needs and does not have ---------------------------
cov = {}
for k in REQUIRED:
    states = {a.get(k, "NOT_CHECKED") for a in arts}
    cov[k] = sorted(str(x) for x in states)
OUT["field_coverage_in_this_limited_sample"] = cov
OUT["scope_limit"] = (
  "This scan covers ONLY _wt_handoff example artifacts with task == small_molecule. It "
  "does not read run directories, training logs, configs, vina caches or source, so it "
  "CANNOT support any claim about whether existing data as a whole is traceable. Where "
  "this scan reports NOT_RECORDED_IN_FIELDS_READ, the field may still be recoverable "
  "elsewhere -- and for the success predicate it demonstrably is: "
  "runs/R3a_*/env_out/vina_cache/cache/*.json carries per-compound status/message/"
  "canonical_smiles/score, and runs/R3a_*/env_out/vina_cache/receptors/ holds "
  "8UN5_A_XQ6_clean.pdb. Per-run recoverability must be settled per run."
)
print(json.dumps(OUT, indent=1, ensure_ascii=False))
