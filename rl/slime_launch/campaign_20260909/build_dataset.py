#!/usr/bin/env python3
"""Consolidate everything this campaign measured into one JSON the notebook reads.

The notebook must not re-derive numbers from logs: figures and prose have to come
from the same file or they will disagree. This script is the only place that parses.
"""
import glob, json, re, collections, statistics as st
W = "/lus/lfs1aip2/projects/public/u6gb/tasks/large-discovery-model/_nanogpt_work"
out = {}

# --- 1. controlled batch-size dose-response (single factor) -------------------
scan = collections.defaultdict(list)
for f in glob.glob(W + "/logs/bs2/*.log") + glob.glob(W + "/logs/scan_*.log"):
    for l in open(f, errors="replace"):
        if not (l.startswith("BS2 TB=") or l.startswith("SCAN TB=")):
            continue
        m = re.search(r"TB=(\d+).*?val_bpb= *([0-9.]+) mfu= *([0-9.]+) steps= *([0-9.]+)", l)
        if not m or "OOM" in l:
            continue
        v = re.search(r"vram_at_claim=(\d+)", l)
        scan[int(m.group(1))].append({
            "val_bpb": float(m.group(2)), "mfu": float(m.group(3)),
            "steps": float(m.group(4)), "vram_at_claim": int(v.group(1)) if v else None})
out["batch_scan"] = {str(k): v for k, v in sorted(scan.items())}

# --- 2. OOM contamination: was the card already occupied? --------------------
oom = []
for f in glob.glob(W + "/logs/bs2/*.log"):
    for l in open(f, errors="replace"):
        if l.startswith("BS2 TB=") and "OOM" in l:
            v = re.search(r"vram_at_claim=(\d+)", l)
            t = re.search(r"TB=(\d+)", l)
            oom.append({"tb": int(t.group(1)), "vram_at_claim": int(v.group(1)) if v else None})
out["oom_cells"] = oom

# --- 3. the reward reference, measured 8x single-tenant ----------------------
out["reference"] = {"val_bpb_mean": 1.003512, "sd": 0.000821, "n": 8,
                    "nodes": ["nid010854","nid010982","nid010968","nid010969",
                              "nid010970","nid010971","nid010972","nid010975"],
                    "note": "8 single-tenant measurements of the default config, "
                            "concurrency 1. Supersedes a contended 1.107706."}

# --- 4. R0c: what the search actually evaluated ------------------------------
r0c = []
for b in glob.glob(W + "/r0c_runs/*/rep*/model_based_buffer.jsonl"):
    rep = b.split("/r0c_runs/")[1].rsplit("/", 1)[0]
    for l in open(b):
        try: r = json.loads(l)
        except Exception: continue
        if isinstance(r.get("score"), (int, float)):
            m = r.get("metrics") or {}
            r0c.append({"rep": rep, "val_bpb": r["score"], "hash": r.get("source_hash"),
                        "params": r.get("params"), "mfu": m.get("mfu_percent"),
                        "steps": m.get("num_steps"), "iteration": r.get("iteration")})
out["r0c"] = r0c

# --- 5. R0a / R0b: the untrained reference rows ------------------------------
rows = []
for f in glob.glob(W + "/r0_results*/*.jsonl"):
    if "deprecated" in f or "seedbug" in f or "noturntext" in f: continue
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        rows.append({k: r.get(k) for k in
                     ("run_label","instance_index","reward","best_val_bpb","n_evaluations",
                      "n_proposals","failure_source","backend","backend_format_identical")})
out["r0ab"] = rows

json.dump(out, open(W + "/campaign_data.json", "w"), indent=1)
print("batch_scan levels :", {k: len(v) for k, v in out["batch_scan"].items()})
print("oom cells         :", len(oom))
print("r0c evaluations   :", len(r0c))
print("r0a/r0b rows      :", len(rows), collections.Counter(r["run_label"] for r in rows))
print("wrote", W + "/campaign_data.json")
