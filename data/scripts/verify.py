#!/usr/bin/env python3
"""
verify.py -- Independent verification for the ldm-2.0 pipeline.

These checks are not decoration: each one was written *because* it caught a real bug
that the scripts happily ran past. Run them after any change to the adapters,
the fabricator, or the renderer.

  coverage : every section of a source prompt reaches the rendered output
             (caught: a 26KB parent train.py silently merged into another section;
              caught: `Current active schema values` parsed but never used, leaving
              the model to make incremental edits without knowing current values)
  validity : every action is executable by the runner
             (caught: training targets taken from REJECTED transcript attempts --
              out-of-range values, over-budget edit counts, wrong types)
  alpaca   : the rendered file is loadable by LlamaFactory
             (checks required fields, dataset_info mapping, JSON-valid outputs,
              contract placement, prompt length vs cutoff_len)
  leakage  : no answer/provenance/legacy-contract text leaks into the prompt, and no
             field's mere presence predicts the action
             (caught: `## Observed history` appearing only on synthetic records, so
              the model could learn the marker instead of the reasoning)

Usage:
    python verify.py coverage --run-dir RUN [--state state_0845]
    python verify.py validity --in-ir ir.jsonl
    python verify.py alpaca   --sft sft.jsonl [--dataset-info dataset_info.json]
                              [--cutoff-len 16384] [--chars-per-token 3.2]
    python verify.py leakage  --sft sft.jsonl --in-ir ir.jsonl
    python verify.py all      --run-dir RUN --in-ir ir.jsonl --sft sft.jsonl \
                              [--dataset-info dataset_info.json]
"""
import argparse, json, sys, collections
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

OK, FAIL, WARN = "  [ok]", "  [FAIL]", "  [warn]"
_state = {"fail": 0, "warn": 0}

def ok(m): print(f"{OK} {m}")
def fail(m): _state["fail"] += 1; print(f"{FAIL} {m}")
def warn(m): _state["warn"] += 1; print(f"{WARN} {m}")

def read_jsonl(p):
    t = Path(p).read_text(encoding="utf-8")
    if t.lstrip().startswith("["):
        return json.loads(t)
    return [json.loads(l) for l in t.splitlines() if l.strip()]

# --------------------------------------------------------------- coverage
def cmd_coverage(a):
    import build_ldm2 as B
    sd = Path(a.run_dir) / "states" / a.state
    if not sd.exists():
        cands = sorted((Path(a.run_dir) / "states").glob("state_*"))
        if not cands:
            fail(f"no states under {a.run_dir}"); return
        sd = cands[len(cands) // 2]
    prompt = (sd / "prompt.md").read_text(encoding="utf-8")
    resp = (sd / "response.md").read_text(encoding="utf-8")
    od = {}
    if (sd / "operations.json").exists():
        od = json.loads((sd / "operations.json").read_text(encoding="utf-8"))

    sec = B._sections(prompt, B.NG_HEADS)
    missing_heads = [h for h in B.NG_HEADS if h not in sec]
    if missing_heads:
        warn(f"headings not found in {sd.name} (bodies merge into the previous "
             f"section): {missing_heads}")
    else:
        ok(f"all {len(B.NG_HEADS)} headings parsed in {sd.name}")

    ir = B.adapt_nanogpt({"instruction": prompt, "output": resp, "operations_doc": od})
    if ir is None:
        fail("adapter returned None"); return
    rendered = B.render_prose(ir)

    # sections we deliberately restate rather than copy verbatim
    RESTATED = {"Active edit operations", "Inactive features available for expansion",
                "Return format", "Current active schema values",
                "Prior operations already applied inside this child state"}
    lost = []
    for h, body in sec.items():
        if h in RESTATED or not body.strip():
            continue
        probe = body.strip().splitlines()[0][:60].strip()
        if probe and probe not in rendered:
            lost.append((h, len(body), probe))
    if lost:
        for h, n, p in lost:
            fail(f"section '{h}' ({n} chars) absent from the rendered prompt: {p!r}")
    else:
        ok("every verbatim-carried section reaches the rendered prompt")

    # restated sections must still be represented, semantically
    ds = ir["search_state"]["design_space"]
    if not ds["active_parameters"]:
        fail("no active parameters parsed")
    elif not any(p.get("current_value") is not None for p in ds["active_parameters"]):
        fail("active parameters carry no `current_value` -- the model cannot know the "
             "baseline it is editing from ('Current active schema values' lost?)")
    else:
        ok("active parameters carry current values")
    if "current=" not in rendered:
        fail("rendered prompt never shows `current=` for any parameter")
    else:
        ok("current values appear in the rendered prompt")
    print(f"       source {len(prompt)} chars -> rendered {len(rendered)} chars")

# --------------------------------------------------------------- validity
def cmd_validity(a):
    irs = read_jsonl(a.in_ir)
    fails = collections.Counter(); ex = {}
    def note(k, e):
        fails[k] += 1; ex.setdefault(k, e)
    for ir in irs:
        ds = ir["search_state"]["design_space"]
        act = ir["action"]
        amap = {p["name"]: p for p in ds["active_parameters"]}
        imap = {p["name"]: p for p in ds["inactive_parameters"]}
        allowed = ir["request"].get("allowed_actions") or []
        maxe = ir["request"].get("max_edits_per_candidate")
        if act["type"] not in allowed:
            note("action_type_not_allowed", f"{act['type']} not in {allowed}")
        if act["type"] == "propose":
            for c in act["payload"].get("candidates", []):
                edits = c.get("edits") or []
                if maxe and len(edits) > maxe:
                    note("too_many_edits", f"{len(edits)} > {maxe}")
                for e in edits:
                    p = amap.get(e.get("parameter"))
                    if p is None:
                        note("param_not_active", str(e.get("parameter"))); continue
                    v = e.get("value")
                    if p.get("type") == "choice":
                        if v not in p["domain"]:
                            note("value_not_in_choice", f"{p['name']}={v!r} vs {p['domain']}")
                    else:
                        try:
                            if not (p["domain"][0] <= v <= p["domain"][1]):
                                note("value_out_of_range", f"{p['name']}={v} vs {p['domain']}")
                        except TypeError:
                            note("value_type_bad", f"{p['name']}={v!r}")
                        if p.get("type") == "int" and not isinstance(v, int):
                            note("int_param_got_float", f"{p['name']}={v!r}")
                    want = "set_choice" if p.get("type") == "choice" else "set_numeric"
                    if e.get("edit_op") != want:
                        note("edit_op_mismatch", f"{p['name']}: {e.get('edit_op')} != {want}")
                    if p.get("current_value") is not None and v == p["current_value"]:
                        note("noop_edit", f"{p['name']}={v} == current")
        elif act["type"] == "expand_design_space":
            t = act["payload"].get("activate")
            if t not in imap:
                note("activate_not_inactive", f"{t} not in {sorted(imap)}")
    if fails:
        for k, v in fails.most_common():
            fail(f"{k}: {v}  e.g. {ex[k]}")
    else:
        ok(f"all {len(irs)} actions are valid against their own design space")

# --------------------------------------------------------------- alpaca
def cmd_alpaca(a):
    recs = read_jsonl(a.sft)
    need = {"instruction", "input", "output"}
    bad = [i for i, r in enumerate(recs)
           if (need - set(r)) or not r.get("instruction") or not r.get("output")]
    if bad:
        fail(f"{len(bad)} records missing/empty alpaca fields (e.g. #{bad[0]})")
    else:
        ok(f"{len(recs)} records carry instruction/input/output")

    nb = 0
    for r in recs:
        try:
            o = json.loads(r["output"])
            assert "type" in o and "payload" in o
        except Exception:
            nb += 1
    if nb: fail(f"{nb} outputs are not valid action JSON")
    else: ok("every output is valid action JSON with type/payload")

    if a.dataset_info:
        info = json.loads(Path(a.dataset_info).read_text(encoding="utf-8"))
        for name, d in info.items():
            if d.get("formatting") != "alpaca":
                fail(f"dataset '{name}': formatting != alpaca")
            exp = {"prompt": "instruction", "query": "input",
                   "response": "output", "system": "system"}
            if d.get("columns") != exp:
                fail(f"dataset '{name}': columns mapping is {d.get('columns')}")
            fn = Path(a.dataset_info).parent / d.get("file_name", "")
            if not fn.exists():
                warn(f"dataset '{name}': file_name '{d.get('file_name')}' not found "
                     f"next to dataset_info.json -- rename it to match what you place "
                     f"in LlamaFactory's data/")
        ok(f"dataset_info registers {len(info)} dataset(s)")

    L = sorted(len(r["instruction"]) for r in recs)
    tok = lambda c: c / a.chars_per_token
    print(f"       instruction chars: median {L[len(L)//2]}  max {L[-1]}")
    print(f"       est. tokens (~{a.chars_per_token} ch/tok): median {tok(L[len(L)//2]):.0f}"
          f"  max {tok(L[-1]):.0f}")
    if tok(L[-1]) > a.cutoff_len:
        fail(f"longest prompt ~{tok(L[-1]):.0f} tok > cutoff_len {a.cutoff_len}: the "
             f"trailing output contract would be truncated away")
    else:
        ok(f"longest prompt fits cutoff_len {a.cutoff_len} (estimate only -- confirm "
           f"with your real tokenizer)")

    tail = sum(1 for r in recs if "Return a single JSON object" in r["instruction"][-600:])
    if tail != len(recs):
        fail(f"{len(recs)-tail} records do not end with the output contract")
    else:
        ok("every prompt ends with the output contract")

# --------------------------------------------------------------- leakage
def cmd_leakage(a):
    recs = read_jsonl(a.sft)
    LEGACY = ["propose_train_operations", "required_output",
              "top-level JSON value must be a list"]
    hits = collections.Counter()
    for r in recs:
        for s in LEGACY:
            if s in r["instruction"]:
                hits[s] += 1
        if "provenance" in r["instruction"] or '"synthetic"' in r["instruction"]:
            hits["provenance"] += 1
    if hits:
        for k, v in hits.items():
            fail(f"'{k}' leaks into {v} prompts")
    else:
        ok("no legacy contract / answer / provenance text in any prompt")

    if not a.in_ir:
        return
    irs = read_jsonl(a.in_ir)
    probes = {
        "observations": lambda ir: bool(ir["search_state"].get("observations")),
        "progress": lambda ir: ir["search_state"].get("progress") is not None,
        "surrogate_feedback": lambda ir: ir["search_state"].get("surrogate_feedback") is not None,
        "do_not_repeat": lambda ir: bool(ir["search_state"].get("do_not_repeat")),
    }
    leaked = False
    for name, fn in probes.items():
        tab = collections.Counter((fn(ir), ir["action"]["type"]) for ir in irs)
        present = {act: n for (p, act), n in tab.items() if p}
        absent = {act for (p, act), n in tab.items() if not p}
        if len(present) == 1 and sum(present.values()) > 10:
            only = next(iter(present))
            if absent - {only}:
                leaked = True
                fail(f"'{name}' present on {sum(present.values())} records, ALL with "
                     f"action='{only}' -- its mere presence predicts the action")
    if not leaked:
        ok("no field's presence predicts the action type")

    viol = 0
    for ir in irs:
        dnr = ir["search_state"].get("do_not_repeat") or []
        if not dnr or ir["action"]["type"] != "propose":
            continue
        excl = {json.dumps(d, sort_keys=True, ensure_ascii=False)
                if isinstance(d, (dict, list)) else str(d) for d in dnr}
        for c in ir["action"]["payload"].get("candidates", []):
            d = c.get("design")
            k = (json.dumps(d, sort_keys=True, ensure_ascii=False)
                 if isinstance(d, (dict, list)) else str(d))
            if k in excl:
                viol += 1; break
    if viol:
        fail(f"{viol} records propose a design their own prompt forbids")
    else:
        ok("no record violates its own do_not_repeat")

# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Verify the ldm-2.0 pipeline.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("coverage"); p.add_argument("--run-dir", required=True)
    p.add_argument("--state", default="state_0845"); p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("validity"); p.add_argument("--in-ir", required=True)
    p.set_defaults(func=cmd_validity)

    p = sub.add_parser("alpaca"); p.add_argument("--sft", required=True)
    p.add_argument("--dataset-info"); p.add_argument("--cutoff-len", type=int, default=16384)
    p.add_argument("--chars-per-token", type=float, default=3.2)
    p.set_defaults(func=cmd_alpaca)

    p = sub.add_parser("leakage"); p.add_argument("--sft", required=True)
    p.add_argument("--in-ir"); p.set_defaults(func=cmd_leakage)

    p = sub.add_parser("all")
    p.add_argument("--run-dir"); p.add_argument("--in-ir", required=True)
    p.add_argument("--sft", required=True); p.add_argument("--dataset-info")
    p.add_argument("--state", default="state_0845")
    p.add_argument("--cutoff-len", type=int, default=16384)
    p.add_argument("--chars-per-token", type=float, default=3.2)
    p.set_defaults(func=None)

    a = ap.parse_args()
    if a.cmd == "all":
        if a.run_dir:
            print("\n--- coverage ---"); cmd_coverage(a)
        print("\n--- validity ---"); cmd_validity(a)
        print("\n--- alpaca ---");   cmd_alpaca(a)
        print("\n--- leakage ---");  cmd_leakage(a)
    else:
        a.func(a)

    print("\n" + "=" * 58)
    if _state["fail"]:
        print(f"RESULT: {_state['fail']} FAIL, {_state['warn']} warn")
        sys.exit(1)
    print(f"RESULT: all checks passed ({_state['warn']} warn)")

if __name__ == "__main__":
    main()
