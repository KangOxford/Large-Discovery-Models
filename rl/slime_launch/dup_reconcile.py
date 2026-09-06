#!/usr/bin/env python3
"""Reconcile B1 / B3 / B4's duplicate counts on ONE fixed runset, one file, one key set.

Scope, stated because the previous audit failed to state one: this reads
`ldm_rl/runs/*/gp_history.jsonl` and, where present, `env_out/vina_cache/cache/*.json`.
It does NOT read the real_80 harness artifact -- B1 (progress.md 380-388) established that
the harness history is produced by different code with different duplicate semantics and is
NOT the same measurement, so it cannot arbitrate this dispute.

Evidence states are explicit: CONFIRMED_FROM_<source> / NOT_RECORDED_IN_FIELDS_READ /
NOT_CHECKED / CONFLICTING. Nothing is declared absent that was never searched for.
"""
import json, glob, hashlib, sys
from pathlib import Path
try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAVE_RDKIT = True
except Exception:
    HAVE_RDKIT = False

T = Path("/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/ldm_rl")
PRIOR = 63

def canon(s):
    if not HAVE_RDKIT: return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None

def rows(gp):
    out = []
    for ln in Path(gp).read_text(errors="replace").splitlines():
        ln = ln.strip()
        if not ln: continue
        try: r = json.loads(ln)
        except Exception: continue
        out.append(r)
    return out

def cost_to_80(smis, seen):
    """rows consumed until 80 distinct keys not already in `seen`."""
    s = set(seen); n = 0
    for i, k in enumerate(smis, 1):
        if k is None: continue
        if k not in s:
            s.add(k); n += 1
            if n == 80: return i
    return None

def cache_status(d):
    c = Path(d)/"env_out/vina_cache/cache"
    if not c.is_dir(): return {"state": "NOT_CHECKED_no_cache_dir"}
    ok = bad = 0; msgs = {}
    ids = set()
    for f in c.glob("*.json"):
        try: j = json.loads(f.read_text())
        except Exception: continue
        st = j.get("status")
        ids.add(j.get("canonical_smiles"))
        if st == "ok": ok += 1
        else:
            bad += 1; msgs[str(st)] = msgs.get(str(st), 0) + 1
    lig = len(list((Path(d)/"env_out/vina_cache/ligands").glob("*"))) \
          if (Path(d)/"env_out/vina_cache/ligands").is_dir() else None
    # NOT the task success predicate. Traced 2026-09-06:
    #   engine_adapters.py:265-289 SmilesCandidateEvaluator.evaluate returns "succeeded"
    #   ONLY when BOTH vina and activity are non-None, else "failed"
    #   ("non-finite objective score after retries");
    #   rl_real_shared.py:98 appends to gp_history only on status == "succeeded".
    # So docking status "ok" is NECESSARY BUT NOT SUFFICIENT: a molecule can dock fine and
    # still fail on the QSAR/activity leg, never reaching gp_history. Label accordingly.
    return {"state": "DOCKING_COMPONENT_STATUS_ONLY_not_task_success_predicate",
            "task_success_predicate": "vina AND activity both non-None (engine_adapters.py:265-289)",
            "entries": ok+bad, "status_ok": ok,
            "status_not_ok": bad, "non_ok_breakdown": msgs,
            "distinct_canonical_in_cache": len(ids - {None}),
            "ligands_prepared": lig,
            # An object-count difference between two directories. NOT a count of failed
            # calls: duplicates, intermediate products, differing cache keys and in-flight
            # work all produce it. Left UNKNOWN until traced.
            "ligands_minus_cache_UNEXPLAINED": (lig - (ok+bad)) if lig is not None else None,
            "ligands_minus_cache_is_NOT_failure_count": True}

sel = sys.argv[1] if len(sys.argv) > 1 else "*"
res = {"scope": "ldm_rl/runs/*/gp_history.jsonl (+ env_out/vina_cache where present)",
       "excluded_source": "real_80 harness artifact -- different code, different dedupe "
                          "semantics (B1 progress.md 380-388); cannot arbitrate this dispute",
       "prior_rows": PRIOR, "rdkit_available": HAVE_RDKIT, "runs": {}}

for d in sorted(glob.glob(str(T/f"runs/{sel}"))):
    d = Path(d); gp = d/"gp_history.jsonl"
    if not gp.exists(): continue
    R = rows(gp)
    if len(R) < PRIOR + 80: continue
    raw = [r.get("smiles") for r in R]
    prior_raw = raw[:PRIOR]; post_raw = raw[PRIOR:]
    e = {"file_sha256_16": hashlib.sha256(gp.read_bytes()).hexdigest()[:16],
         "total_rows": len(R), "post_prior_rows": len(post_raw),
         "prior_distinct_raw": len(set(x for x in prior_raw if x))}
    # B4 convention: 80 unique from POST-PRIOR, prior NOT seeded as seen
    e["B4_post_prior_no_seed_raw"] = cost_to_80(post_raw, set())
    # B1/B3 convention: 80 unique NEW, prior SMILES seeded as seen
    e["B1_post_prior_seeded_raw"] = cost_to_80(post_raw, set(x for x in prior_raw if x))
    # B3's other convention: from FILE START
    e["B3_file_start_raw"] = cost_to_80(raw, set())
    if HAVE_RDKIT:
        cp = [canon(x) if x else None for x in prior_raw]
        cq = [canon(x) if x else None for x in post_raw]
        e["B4_post_prior_no_seed_canon"] = cost_to_80(cq, set())
        e["B1_post_prior_seeded_canon"] = cost_to_80(cq, set(x for x in cp if x))
    e["success_predicate"] = cache_status(d)
    res["runs"][d.name] = e

print(json.dumps(res, indent=1, ensure_ascii=False))
