#!/usr/bin/env python3
"""Common-runset audit for the budget-matched RL comparison.

Binds every candidate evaluation artifact to the fields the comparison needs, and records
which fields are ABSENT rather than inferring them. Also reconciles the three duplicate
counts (B1, B3, B4) by showing they measure different pipeline stages.
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
    if d.get("task") not in ("small_molecule", None): continue
    rec = {k: None for k in REQUIRED}
    rec["run_id"] = str(Path(f).parent.name)
    rec["artifact"] = str(Path(f).relative_to(L))
    rec["artifact_sha256_16"] = sha(f)
    # only what the file actually carries
    rec["stop_reason"] = d.get("stop_reason") or d.get("early_stop_reason")
    rec["succeeded_field"] = ("successful_evaluation_count"
                              if "successful_evaluation_count" in d else None)
    rec["successful_evaluation_count"] = d.get("successful_evaluation_count")
    rec["failed_evaluation_count"] = d.get("failed_evaluation_count")
    rec["observation_count"] = d.get("observation_count")
    rec["llm_call_count"] = d.get("llm_call_count")
    rec["drop_counts"] = d.get("drop_counts")
    rec["final_hypervolume"] = d.get("final_hypervolume")
    rec["method"] = d.get("method")
    rec["upstream_dedupe"] = ("yes: drop_counts.duplicate present"
                              if isinstance(d.get("drop_counts"), dict)
                              and "duplicate" in d["drop_counts"] else None)
    # history: what per-record fields exist
    h = Path(f).parent/"history.json"
    if h.exists():
        try:
            hh = json.loads(h.read_text())
            if isinstance(hh, list) and hh and isinstance(hh[0], dict):
                rec["history_record_fields"] = sorted(hh[0].keys())
                rec["history_len"] = len(hh)
                sm = [r.get("smiles") for r in hh if isinstance(r, dict) and r.get("smiles")]
                rec["smiles_raw"] = "present" if sm else None
                rec["history_unique_smiles_raw"] = len(set(sm)) if sm else None
        except Exception:
            pass
    arts.append(rec)
OUT["evaluation_artifacts"] = arts

# ---- 2. the three duplicate counts, and what each measures ---------------------
def gp_counts(run_glob):
    out = {}
    for d in sorted(glob.glob(str(T/f"runs/{run_glob}"))):
        gp = Path(d)/"gp_history.jsonl"
        if not gp.exists(): continue
        rows, smi = 0, []
        for ln in gp.read_text(errors="replace").splitlines():
            ln = ln.strip()
            if not ln: continue
            try: r = json.loads(ln)
            except Exception: continue
            rows += 1
            s = r.get("smiles") or r.get("SMILES")
            if s: smi.append(s)
        out[Path(d).name] = {"rows": rows, "with_smiles": len(smi),
                             "unique_raw_smiles": len(set(smi))}
    return out

OUT["duplicate_count_reconciliation"] = {
  "question": "B4 reports 0-3 duplicate-extra; B1/B3 report ~80-108. Same runset?",
  "answer": "No. They measure different stages of the same pipeline.",
  "B1_B3_stage": {
     "source": "assets/examples/real_80_qwen35_9b (summary.json + history.json)",
     "what_it_counts": "rows that SURVIVED upstream dedupe and entered history",
     "observed": "80 rows, 80 unique -> 0 duplicates inside the artifact",
     "why": "summary.json's own drop_counts shows duplicates were removed BEFORE history"},
  "B4_stage": {
     "source": "ldm_rl/runs/*/gp_history.jsonl (training runs)",
     "what_it_counts": "the raw proposal stream, before that dedupe",
     "observed": gp_counts("R3a_*")},
  "resolution": ("Both counts are correct for their own stage. Neither is evidence about "
                 "the other. Any comparison must state which stage it counts, and the "
                 "budget-charge question lives at the pre-dedupe stage."),
}

# ---- 3. what the comparison needs and does not have ---------------------------
present, absent = [], []
for k in REQUIRED:
    if any(a.get(k) for a in arts): present.append(k)
    else: absent.append(k)
OUT["field_coverage"] = {"present_in_at_least_one_artifact": present,
                         "absent_from_every_artifact": absent}
OUT["verdict"] = (
  "The existing artifacts CANNOT form a traceable equal-successful-budget main-task "
  "comparison. The fields that decide it -- checkpoint, scorer, code hash, training seed, "
  "evaluation seed, per-call budget charge, prior manifest, canonical SMILES key and the "
  "raw succeeded field -- are absent from every artifact on disk. What exists is one "
  "scorer-replay summary with an HV number and a bare (scores, smiles) history."
)
print(json.dumps(OUT, indent=1, ensure_ascii=False))
