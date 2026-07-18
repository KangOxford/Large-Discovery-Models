#!/usr/bin/env python3
"""
build_ldm2.py -- Convert heterogeneous LDM task traces into the unified `ldm-2.0`
intermediate representation, then render to LlamaFactory alpaca format.

Pipeline:
    raw task trace --[adapter]--> ldm-2.0 IR --[renderer]--> LlamaFactory alpaca

Design notes (see FORMAT_ldm-2.0.md):
  * The design space is STATE (search_state.design_space), not a static header,
    so the model can learn to EXPAND it, not merely fill it in.
  * `action` is a tagged union: propose | expand_design_space | add_new_parameter.
  * Language first: every layer carries `description`; the action carries
    `reasoning`. Structure is scaffolding; meaning lives in prose.
  * Missing data is explicit (null + flag), never fabricated.

Usage:
    # sample file containing {smallmol, nanogpt, protein} lists
    python build_ldm2.py from-sample --in ldm_data_sample.json --out-ir ir.jsonl

    # a full nanogpt run directory (states/state_*/)
    python build_ldm2.py from-nanogpt-run --run-dir ./expanded_ldm_bon_N4H4_03 \
        --out-ir ir_nanogpt.jsonl --min-status evaluated

    # IR -> LlamaFactory
    python build_ldm2.py render --in-ir ir.jsonl --out sft.jsonl \
        --render prose --dataset-info dataset_info.json

    # audit action-type / parameter distribution (mode-collapse check)
    python build_ldm2.py audit --in-ir ir.jsonl
"""
import argparse, json, re, glob, collections, os
from pathlib import Path

SCHEMA_VERSION = "ldm-2.0"

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def jdump(o, indent=None):
    return json.dumps(o, ensure_ascii=False, indent=indent)

def write_jsonl(recs, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(jdump(r) + "\n")

def read_jsonl(path):
    txt = Path(path).read_text(encoding="utf-8")
    if txt.lstrip().startswith("["):
        return json.loads(txt)
    return [json.loads(l) for l in txt.splitlines() if l.strip()]

def _sections(text, heads, warn_name=None):
    """Split a markdown-ish instruction into {head: body} by known headings.

    Headings must be given EXACTLY as they appear (e.g. "Current parent `train.py`"
    includes backticks). A near-miss silently merges that section's body into the
    preceding one -- which once swallowed a 27KB train.py into `Return format`. The
    coverage check below makes that failure loud instead of silent.
    """
    idx = []
    for h in heads:
        i = text.find(h + ":")
        if i >= 0:
            idx.append((i, h))
    idx.sort()
    out = {}
    for n, (i, h) in enumerate(idx):
        j = idx[n + 1][0] if n + 1 < len(idx) else len(text)
        out[h] = text[i + len(h) + 1:j].strip()
    if warn_name:
        missed = [h for h in heads if h not in out]
        if missed:
            print(f"[warn] {warn_name}: headings not found (their bodies will be "
                  f"merged into the preceding section): {missed}")
    return out

# ----------------------------------------------------------------------------
# ADAPTER 1: nanogpt (training-program search)
# ----------------------------------------------------------------------------
NG_HEADS = ['Search state', 'Project and benchmark context', 'Objective',
            'Active edit operations', 'Inactive features available for expansion',
            'Current active schema values', 'Prior operations already applied inside this child state',
            'Recent real evaluated states', 'Feedback from previous iterations',
            'Recent evaluated trials', 'Return format', 'Current parent `train.py`']

_PARAM_NUM = re.compile(r'^-\s*(\w+):\s*(int|float)\s+in\s+\[([^\]]+)\](?:\s*scale=(\w+))?')
_PARAM_CHO = re.compile(r'^-\s*(\w+):\s*choice\s+in\s+\[([^\]]+)\]')
_INACT     = re.compile(r'^-\s*(\w+):\s*(?:(int|float)\s+in\s+\[([^\]]+)\](?:\s*scale=(\w+))?|choice\s+in\s+\[([^\]]+)\])\s*;\s*current=(\S+)')

def _lit(s):
    s = s.strip().strip("'\"")
    for cast in (int, float):
        try: return cast(s)
        except ValueError: pass
    return s

def _parse_active(block):
    params = []
    for line in (block or '').splitlines():
        line = line.strip()
        m = _PARAM_CHO.match(line)
        if m:
            params.append({"name": m.group(1), "type": "choice",
                           "domain": [_lit(x) for x in m.group(2).split(',')],
                           "edit_op": "set_choice"})
            continue
        m = _PARAM_NUM.match(line)
        if m:
            lo, hi = [_lit(x) for x in m.group(3).split(',')]
            p = {"name": m.group(1), "type": m.group(2), "domain": [lo, hi],
                 "edit_op": "set_numeric"}
            if m.group(4): p["scale"] = m.group(4)
            params.append(p)
    return params

def _parse_inactive(block):
    out = []
    for line in (block or '').splitlines():
        m = _INACT.match(line.strip())
        if not m: continue
        name, typ, rng, scale, choices, cur = m.groups()
        if choices is not None:
            p = {"name": name, "type": "choice",
                 "domain": [_lit(x) for x in choices.split(',')]}
        else:
            lo, hi = [_lit(x) for x in rng.split(',')]
            p = {"name": name, "type": typ, "domain": [lo, hi]}
            if scale: p["scale"] = scale
        p["current_value"] = _lit(cur)
        out.append(p)
    return out

def _parse_ng_output(output, accepted_ops=None, max_ops=None):
    """Return (action_type, payload, reasoning, summary) for the ACCEPTED call.

    response.md is a transcript that can contain REJECTED attempts before the
    accepted one, e.g. validation_log:
        attempt 1 rejected: EMBEDDING_LR=0.05 outside [0.1, 1.0]
        attempt 2 rejected: EMBEDDING_LR=0.075 outside [0.1, 1.0]
        attempt 3 accepted
    Taking the first tool call would train the model to emit actions the runner
    rejects (out-of-range values, over-limit edit counts). We therefore prefer the
    call matching operations.json's accepted set, and otherwise fall back to the
    LAST valid call rather than the first.
    """
    m = re.search(r'\[.*\]', output, re.S)
    if not m:
        return None, None, None, None
    try:
        calls = json.loads(m.group(0))
    except Exception:
        return None, None, None, None

    flat = []
    for entry in calls:
        for tc in entry.get('tool_calls', []):
            flat.append(tc)
    if not flat:
        return None, None, None, None

    def _mk(tc):
        name, args = tc.get('name'), (tc.get('arguments') or {})
        if name == 'propose_train_operations':
            ops = args.get('operations', []) or []
            edits = [{"parameter": o.get('name'), "edit_op": o.get('op'),
                      "value": o.get('value'), "rationale": o.get('rationale')}
                     for o in ops]
            reasoning = " ".join(o.get('rationale', '') for o in ops if o.get('rationale')).strip()
            return ("propose", {"candidates": [{"parent": None, "edits": edits}]},
                    reasoning or None, args.get('summary'))
        if name == 'propose_operation_feature':
            feat = args.get('name') or args.get('feature') or args.get('parameter')
            payload = {"activate": feat}
            if 'value' in args:
                payload["initial_value"] = args['value']
            return ("expand_design_space", payload,
                    args.get('rationale') or args.get('reason'), args.get('summary'))
        return None

    # 1) exact match against the accepted operation set from operations.json.
    #    Compare types too: str(64) == str('64'), so a type-blind key would accept a
    #    REJECTED attempt that proposed HEAD_DIM='64' as if it were the accepted
    #    HEAD_DIM=64. The runner records only correctly-typed values as accepted.
    def _key(o):
        v = o.get('value')
        return (o.get('name'), type(v).__name__, str(v))

    if accepted_ops:
        want = sorted(_key(o) for o in accepted_ops)
        for tc in flat:
            if tc.get('name') != 'propose_train_operations':
                continue
            got = sorted(_key(o)
                         for o in ((tc.get('arguments') or {}).get('operations') or []))
            if got == want:
                return _mk(tc)

    # 2) no transcript call matches the accepted set -> trust the runner's validated
    #    record itself rather than guessing at a transcript entry that may be a
    #    rejected attempt.
    if accepted_ops:
        edits = [{"parameter": o.get('name'), "edit_op": o.get('op'),
                  "value": o.get('value'), "rationale": o.get('rationale')}
                 for o in accepted_ops]
        if edits:
            reasoning = " ".join(o.get('rationale', '') for o in accepted_ops
                                 if o.get('rationale')).strip()
            summary = None
            for tc in flat:
                s = (tc.get('arguments') or {}).get('summary')
                if s:
                    summary = s
            return ("propose", {"candidates": [{"parent": None, "edits": edits}]},
                    reasoning or None, summary)

    # 3) last resort (e.g. feature-expansion calls, which operations.json does not
    #    describe as `operations`): the LAST call respecting the edit budget.
    for tc in reversed(flat):
        r = _mk(tc)
        if not r:
            continue
        if r[0] == 'propose' and max_ops:
            n = len(r[1]['candidates'][0]['edits'])
            if n > max_ops or n == 0:
                continue
        return r
    return None, None, None, None

_TRIAL = re.compile(
    r'iter=(\d+)\s+(?:selected|warmup)\s+(state_\d+)\s+status=(\w+)\s+val_bpb=([\d.]+)')
_SURR = re.compile(r'surrogate_pred=([\d.]+)\s+std=([\d.]+)\s+score=([\d.]+)')
_ACT = re.compile(r'action=(.+)$')
_RECENT = re.compile(r'-\s*(state_\d+):\s*val_bpb=([\d.]+)[^\n]*?status=(\w+)[^\n]*?note=([^\n]+)')

def _parse_ng_observations(sec):
    """Turn the prose history into structured observations.

    Without this, real nanogpt records carry observations=[] while F1's synthetic
    records carry 18 -- so the rendered '## Observed history' heading appears ONLY on
    synthetic records and perfectly predicts expand_design_space. The model would
    learn that shortcut instead of the stall/exhaustion reasoning. Keeping both real
    and synthetic records structurally identical is what makes the augmentation work.

    Parsed line-by-line on purpose: a lazy `[^\\n]*?` in front of an OPTIONAL group
    will always skip that group, which silently dropped every surrogate_pred.
    """
    obs, seen = [], set()
    for m in _RECENT.finditer(sec.get('Recent real evaluated states', '') or ''):
        sid, vb, st, note = m.groups()
        if sid in seen:
            continue
        seen.add(sid)
        obs.append({"design": {"state": sid, "action": note.strip()},
                    "results": {"val_bpb": float(vb)},
                    "roles": ["evaluated"], "description": f"status={st}"})
    for line in (sec.get('Recent evaluated trials', '') or '').splitlines():
        m = _TRIAL.search(line)
        if not m:
            continue
        it, sid, st, vb = m.groups()
        if sid in seen:
            continue
        seen.add(sid)
        a = _ACT.search(line)
        o = {"design": {"state": sid, "action": a.group(1).strip() if a else None},
             "results": {"val_bpb": float(vb)},
             "roles": ["recent"], "round": int(it), "description": f"status={st}"}
        s = _SURR.search(line)
        if s:
            o["surrogate"] = {"predicted_mean": float(s.group(1)),
                              "uncertainty": float(s.group(2)),
                              "acquisition_value": float(s.group(3))}
        obs.append(o)
    return obs

def _parse_json_block(block):
    """Extract the first ```json ... ``` payload from a section body."""
    if not block:
        return None
    m = re.search(r'```json\s*(.*?)```', block, re.S)
    if not m:
        m = re.search(r'(\{.*\}|\[.*\])', block, re.S)
        if not m:
            return None
        raw = m.group(1)
    else:
        raw = m.group(1)
    try:
        return json.loads(raw)
    except Exception:
        return None

def adapt_nanogpt(rec):
    ins = rec['instruction']
    sec = _sections(ins, NG_HEADS)
    active = _parse_active(sec.get('Active edit operations'))
    inactive = _parse_inactive(sec.get('Inactive features available for expansion'))

    # `Current active schema values` holds each parameter's CURRENT value. Dropping it
    # (as an earlier version did) leaves the model editing blind: it cannot know
    # HEAD_DIM is currently 128 when asked to set it to 96, and the real inference
    # prompt does carry this block -- so losing it is also a train/inference mismatch.
    cur = _parse_json_block(sec.get('Current active schema values')) or {}
    for p in active:
        if p['name'] in cur:
            p['current_value'] = cur[p['name']]

    # Within a multi-pass transition this lists what has already been applied to the
    # child state; without it the model cannot tell which edits are still pending.
    prior_ops = _parse_json_block(
        sec.get('Prior operations already applied inside this child state'))

    # operations.json carries the runner's VALIDATED result: the accepted operation
    # set and the real edit budget. Without it we cannot tell an accepted call from a
    # rejected attempt in the response transcript.
    ops_doc = rec.get('operations_doc') or {}
    accepted_ops = ops_doc.get('operations')
    max_ops = ops_doc.get('max_operations_per_step') or 2

    atype, payload, reasoning, summary = _parse_ng_output(rec['output'], accepted_ops, max_ops)
    if atype is None:
        return None

    # round / parent from the search-state block
    ss = sec.get('Search state', '')
    m = re.search(r'Parent state:\s*state_(\d+)', ss)
    rnd = int(m.group(1)) if m else None
    if atype == "propose" and rnd is not None:
        payload["candidates"][0]["parent"] = f"state_{rnd:04d}"

    fb = sec.get('Feedback from previous iterations', '')
    mbest = re.search(r'val_bpb=([\d.]+)', fb)
    best = ({"design": None, "results": {"val_bpb": float(mbest.group(1))},
             "description": fb.strip()[:400]} if mbest else None)

    allowed = ["propose"] + (["expand_design_space"] if inactive else [])

    obs = _parse_ng_observations(sec)
    # Spec section 3.4: nanogpt DOES expose surrogate feedback (surrogate_pred/std/score in
    # the trial prose). Surface the most recent one structurally instead of leaving
    # the field null while the data sits in prose.
    sfb = None
    for o in reversed(obs):
        if o.get('surrogate'):
            sfb = dict(o['surrogate'])
            sfb['description'] = (f"Surrogate prediction for the most recently selected candidate "
                                  f"({o['design'].get('state')}): "
                                  f"mean={sfb['predicted_mean']}, std={sfb['uncertainty']}, "
                                  f"acquisition={sfb['acquisition_value']}")
            break

    # Structural parity with synthetic records matters as much as content: if
    # `progress` / `do_not_repeat` appear ONLY on fabricated records, their rendered
    # headings alone predict expand_design_space and the model learns the marker
    # instead of the reasoning. Derive both from the real trace.
    since = None
    vals = [(o['round'], o['results']['val_bpb']) for o in obs
            if o.get('round') is not None and o['results']['val_bpb'] < 1e8]
    if vals:
        vals.sort()
        best_i, best_v = 0, vals[0][1]
        for i, (_, v) in enumerate(vals):
            if v < best_v - 1e-9:
                best_v, best_i = v, i
        since = len(vals) - 1 - best_i
    progress = None
    if since is not None:
        progress = {
            "stalled": since >= 5,
            "rounds_since_improvement": since,
            "description": (f"Across the latest {len(vals)} real evaluations, the best value "
                            f"occurred {since} round(s) ago."
                            if since else "The most recent round just improved the best value."),
        }
    # NOTE: nanogpt has NO hard exclusion constraint in the source prompt (unlike
    # protein's do_not_repeat / smallmol's avoid_exact_smiles, which are real).
    # Synthesising one here would put "Do not repeat: [...]" in the prompt while the
    # teacher's target action does exactly that -- training the model to disobey its
    # own instructions. The exhaustion evidence lives in `observations` + `progress`
    # instead, which both real and synthetic records carry (so no structural shortcut).
    dnr = []

    return {
        "schema_version": SCHEMA_VERSION,
        "task": {
            "id": "nanogpt", "domain": "training_program",
            "description": (sec.get('Project and benchmark context') or '').strip(),
            "objectives": [{"name": "val_bpb", "direction": "minimize",
                            "description": (sec.get('Objective') or '').strip()}],
            "reasoning_available": True,
        },
        "search_state": {
            "round": rnd, "num_evaluated": None,
            "design_space": {
                "representation": "parameter_edits",
                "active_parameters": active,
                "inactive_parameters": inactive,
                "expansion_history": [],
                "allows_new_parameters": True,
                "applied_this_transition": prior_ops or [],
                "description": ("The surrogate observes only active parameters; activating an "
                                "inactive parameter expands the feature vector for later candidates."),
            },
            "observations": obs,
            "best_so_far": best,
            "surrogate_feedback": sfb,
            "progress": progress,
            "do_not_repeat": dnr,
        },
        "request": {
            "allowed_actions": allowed,
            "num_candidates": 1,
            "max_edits_per_candidate": max_ops,
            # NOTE: the source 'Return format' section describes the legacy tool-call
            # contract (`propose_train_operations` / `propose_operation_feature`).
            # ldm-2.0 replaces that contract with a single JSON action, so we restate
            # it here rather than copying prose that would contradict the target output.
            "description": (
                "Choose exactly one action.\n"
                "- `propose`: stay in the current design space and edit train.py "
                "(1-2 edits, each on an active parameter, values inside its domain).\n"
                "- `expand_design_space`: activate one inactive parameter. Use this when "
                "the active feature set is too narrow for the next useful search move.\n"
                "Use edit_op=set_numeric for int/float parameters and set_choice for "
                "choice parameters. Do not output diffs or SEARCH/REPLACE blocks; the "
                "runner applies valid edits deterministically."
            ),
        },
        "action": {"type": atype, "reasoning": reasoning,
                   "payload": payload, "summary": summary},
        # fidelity escape hatch: keep the original prose we did not model
        "raw_context": {
            "legacy_return_format": (sec.get('Return format') or '').strip(),
            "search_state_note": ss.strip(),
            "feedback": fb.strip(),
            "recent_real_evaluated": (sec.get('Recent real evaluated states') or '').strip(),
            "recent_trials": (sec.get('Recent evaluated trials') or '').strip(),
            "parent_train_py": (sec.get('Current parent `train.py`') or '').strip(),
        },
    }

# ----------------------------------------------------------------------------
# ADAPTER 2: smallmol (multi-objective molecule design)
# ----------------------------------------------------------------------------
SM_HEADS = ['Task', 'Target context', 'Background', 'Molecule context table',
            'How to use the molecule context', 'Generation principles',
            'SMILES hygiene', 'Generation focus', 'History summary',
            'JSON output format']

ROLE_MAP = {
    "pareto_front": "pareto_front",
    "top_low_vina": "top_objective_0",
    "top_high_activity": "top_objective_1",
    "balanced_elites": "elite",
    "recent_selected": "recent",
}

# Sentences describing the LEGACY output contract. ldm-2.0 supplies its own contract
# via the renderer, so leaving these in would put contradictory instructions in the
# prompt (e.g. "top-level JSON must be a list of strings" vs the action envelope).
_FORMAT_PROSE = re.compile(
    r'(Use compact minified JSON[^.]*\.'
    r'|Return JSON only[^.]*\.'
    r'|The top-level JSON value[^.]*\.'
    r'|Do not include ids, scores[^.]*\.'
    r'|keep each rationale under \d+ words\.?)', re.I)

def _strip_format_prose(text):
    return re.sub(r'\s{2,}', ' ', _FORMAT_PROSE.sub('', text or '')).strip()

def adapt_smallmol(rec):
    ins = rec['instruction']
    sec = _sections(ins, SM_HEADS)
    try:
        hist = json.loads(sec.get('History summary', '{}'))
    except Exception:
        hist = {}
    try:
        ctx = json.loads(sec.get('Molecule context table', '[]'))
    except Exception:
        ctx = []

    # merge the 5 overlapping history views into role-tagged observations
    obs = {}
    for view, role in ROLE_MAP.items():
        for e in hist.get(view, []) or []:
            smi = e.get('smiles')
            o = obs.setdefault(smi, {"design": smi, "results": None, "roles": []})
            sc = e.get('scores')
            if sc and o["results"] is None:
                o["results"] = {"vina": sc[0], "activity": sc[1]}
            if role not in o["roles"]:
                o["roles"].append(role)
    # attach qualitative descriptors from the molecule context table
    for c in ctx:
        smi = c.get('smiles')
        if smi in obs:
            obs[smi]["description"] = "; ".join(
                f"{k}={v}" for k, v in c.items()
                if k not in ('smiles',) and v is not None)

    try:
        out = json.loads(rec['output'])
        cands = [{"design": e.get('smiles'), "rationale": e.get('rationale')}
                 for e in out.get('direct_smiles', [])]
    except Exception:
        return None

    alert = hist.get('recent_diversity_alert')
    progress = ({"stalled": True, "rounds_since_improvement": None,
                 "description": (alert or {}).get('instruction')} if alert else None)

    return {
        "schema_version": SCHEMA_VERSION,
        "task": {
            "id": "smallmol", "domain": "molecule",
            "description": (sec.get('Target context') or '').strip() + "\n\n"
                           + (sec.get('Background') or '').strip(),
            "objectives": [
                {"name": "vina_docking", "direction": "minimize",
                 "description": "AutoDock Vina docking score; lower is better."},
                {"name": "neural_activity", "direction": "maximize",
                 "description": "Target-specific activity model prediction; higher is better."},
            ],
            "reasoning_available": True,
        },
        "search_state": {
            "round": None,
            "num_evaluated": hist.get('n_evaluated'),
            "design_space": {
                "representation": "complete_design",
                "active_parameters": [],
                "inactive_parameters": [],
                "expansion_history": [],
                "allows_new_parameters": True,
                "description": (sec.get('SMILES hygiene') or '').strip(),
            },
            "observations": list(obs.values()),
            "best_so_far": None,
            "surrogate_feedback": None,
            "progress": progress,
            "do_not_repeat": hist.get('avoid_exact_smiles', []) or [],
        },
        "request": {
            "allowed_actions": ["propose"],
            "num_candidates": len(cands) or 8,
            "max_edits_per_candidate": None,
            # The source 'Task' section mixes domain guidance with the legacy output
            # contract ("Use compact minified JSON ..."). Strip the format sentences;
            # the ldm-2.0 contract is supplied by the renderer. Keep the domain focus.
            "description": _strip_format_prose(
                (sec.get('Task') or '').strip() + "\n"
                + (sec.get('Generation focus') or '').strip()),
        },
        "action": {"type": "propose", "reasoning": None,
                   "payload": {"candidates": cands}, "summary": None},
        "raw_context": {
            "generation_principles": (sec.get('Generation principles') or '').strip(),
            "how_to_use_context": (sec.get('How to use the molecule context') or '').strip(),
        },
    }

# ----------------------------------------------------------------------------
# ADAPTER 3: protein (antibody CDRH3 design)
# ----------------------------------------------------------------------------
def adapt_protein(rec):
    try:
        ins = json.loads(rec['instruction'])
    except Exception:
        return None
    cons = ins.get('constraints', {}) or {}
    hist = ins.get('history', {}) or {}

    obs = {}
    for e in hist.get('best', []) or []:
        s = e.get('sequence')
        obs.setdefault(s, {"design": s, "results": {"binding_energy": e.get('score')},
                           "roles": []})["roles"].append("best")
    for e in hist.get('recent', []) or []:
        s = e.get('sequence')
        o = obs.setdefault(s, {"design": s,
                               "results": {"binding_energy": e.get('score')}, "roles": []})
        if "recent" not in o["roles"]: o["roles"].append("recent")
        if e.get('iter') is not None: o["round"] = e['iter']

    try:
        cands = [{"design": s, "rationale": None} for s in json.loads(rec['output'])]
    except Exception:
        return None

    # NOTE: `required_output` is deliberately DROPPED -- in the source data it is a
    # stale constant that leaks the answer on the first record and is simply wrong
    # on the rest. Keeping it would teach the model to ignore its instructions.
    alphabet = cons.get('alphabet')
    length = cons.get('length')
    active = []
    if length and alphabet:
        active = [{"name": "sequence", "type": "string",
                   "domain": {"length": length, "alphabet": list(alphabet)},
                   "edit_op": None}]

    return {
        "schema_version": SCHEMA_VERSION,
        "task": {
            "id": "protein", "domain": "antibody_sequence",
            "description": (ins.get('task', '') + " " +
                            ins.get('important_difference_from_LDM', '')).strip(),
            "objectives": [{"name": "binding_energy", "direction": "minimize",
                            "description": ins.get('objective')}],
            "reasoning_available": False,   # source trace carries no rationale
        },
        "search_state": {
            "round": None,
            "num_evaluated": hist.get('num_observed'),
            "design_space": {
                "representation": "complete_design",
                "active_parameters": active,
                "inactive_parameters": [],
                "expansion_history": [],
                "allows_new_parameters": False,
                "description": "Sequence length and alphabet are fixed; developability filters are listed in constraints.",
            },
            "observations": list(obs.values()),
            "best_so_far": (lambda b: {"design": b[0].get('sequence'),
                                       "results": {"binding_energy": b[0].get('score')}}
                            if b else None)(hist.get('best') or []),
            "surrogate_feedback": None,
            "progress": None,
            "do_not_repeat": cons.get('do_not_repeat', []) or [],
        },
        "request": {
            "allowed_actions": ["propose"],
            "num_candidates": cons.get('num_sequences', 1),
            "max_edits_per_candidate": None,
            # NOTE: the source `output_rules` describe the legacy contract ("top-level
            # JSON must be a list of strings", "no rationales"), which directly
            # contradicts the ldm-2.0 action envelope. We restate the DOMAIN rules
            # only; the output contract is supplied by the renderer.
            "description": (
                f"Propose {cons.get('num_sequences', 1)} CDRH3 sequence(s) of length "
                f"{cons.get('length')} over the alphabet {cons.get('alphabet')}. "
                "Sequences must be novel relative to do_not_repeat, and should respect "
                "the developability filter (see target context)."
            ),
        },
        "action": {"type": "propose", "reasoning": None,
                   "payload": {"candidates": cands}, "summary": None},
        "raw_context": {
            "legacy_output_rules": ins.get('output_rules', []) or [],
            "target_id": ins.get('antigen'),
            "target_context": (ins.get('antigen_context') or {}),
            "developability_filter": cons.get('developability_filter_used_by_code'),
        },
    }

ADAPTERS = {"nanogpt": adapt_nanogpt, "smallmol": adapt_smallmol, "protein": adapt_protein}

# ----------------------------------------------------------------------------
# RENDERER: ldm-2.0 -> LlamaFactory alpaca
# ----------------------------------------------------------------------------
SYSTEM_BY_TASK = {
    "nanogpt": ("You are a scientific search agent proposing edits to a training "
                "program under an iterative model-based (Bayesian) optimization loop. "
                "You may either propose edits within the current design space or expand "
                "that space. Return ONLY the JSON action. Never predict objective values, "
                "surrogate mean/variance, acquisition, or rank."),
    "smallmol": ("You are a scientific search agent proposing candidate molecules under "
                 "an iterative multi-objective Bayesian optimization loop. Return ONLY the "
                 "JSON action. Never predict docking score, activity, EHVI, uncertainty, or rank."),
    "protein": ("You are a scientific search agent proposing candidate antibody CDRH3 "
                "sequences under an iterative Bayesian optimization loop. Return ONLY the "
                "JSON action. Never predict binding energy, uncertainty, or rank."),
}
GENERIC_SYSTEM = ("You are a scientific search agent in an iterative model-based "
                  "(Bayesian) optimization loop. Given the task, the current design space, "
                  "and the observed history, choose one action. Return ONLY the JSON action. "
                  "Never predict objective values, surrogate statistics, or rank.")

def _fmt_params(params):
    out = []
    for p in params:
        d = p.get('domain')
        if isinstance(d, dict):
            dom = f"length={d.get('length')}, alphabet={''.join(d.get('alphabet', []))}"
        elif p.get('type') == 'choice':
            dom = f"choice in {d}"
        else:
            dom = f"{p.get('type')} in {d}" + (f", scale={p['scale']}" if p.get('scale') else "")
        line = f"- {p['name']}: {dom}"
        if p.get('edit_op'): line += f"; edit_op={p['edit_op']}"
        if p.get('current_value') is not None: line += f"; current={p['current_value']}"
        out.append(line)
    return "\n".join(out) if out else "(none)"

def render_prose(ir, include_parent_artifact=True):
    """Render task+search_state+request as natural language + embedded JSON."""
    t, s, r = ir['task'], ir['search_state'], ir['request']
    ds = s['design_space']
    P = []

    P.append(f"# Task: {t['id']} ({t['domain']})\n{t.get('description','').strip()}")

    obj = "\n".join(f"- {o['name']}: {o['direction']} - {o.get('description','')}"
                    for o in t['objectives'])
    P.append(f"## Objectives\n{obj}")

    space = [f"Representation: {ds['representation']}",
             f"\nActive parameters (the surrogate models only these; `current` is the "
             f"value in the parent artifact right now):\n{_fmt_params(ds['active_parameters'])}"]
    if ds['inactive_parameters']:
        space.append("\nInactive parameters (available to activate; activating one expands "
                     "the surrogate's feature space for later candidates):\n"
                     + _fmt_params(ds['inactive_parameters']))
    if ds.get('applied_this_transition'):
        space.append("\nAlready applied inside this child state (do not re-apply):\n"
                     + jdump(ds['applied_this_transition'], indent=1))
    if ds.get('expansion_history'):
        space.append("\nExpansion history:\n" + "\n".join(
            f"- round {e.get('round')}: activated {e.get('activated')} ({e.get('reason')})"
            for e in ds['expansion_history']))
    space.append(f"\nNew parameters may be invented: {ds['allows_new_parameters']}")
    if ds.get('description'): space.append(f"\n{ds['description']}")
    P.append("## Design space (current state - you may act on it)\n" + "\n".join(space))

    if s.get('observations'):
        P.append("## Observed history\n" + jdump(s['observations'], indent=1))
    if s.get('num_evaluated') is not None:
        P.append(f"Evaluations so far: {s['num_evaluated']}")
    if s.get('best_so_far'):
        P.append("## Best so far\n" + jdump(s['best_so_far'], indent=1))
    if s.get('surrogate_feedback'):
        P.append("## Surrogate feedback\n" + jdump(s['surrogate_feedback'], indent=1))
    if s.get('progress'):
        P.append("## Progress\n" + jdump(s['progress'], indent=1))
    if s.get('do_not_repeat'):
        P.append("## Do not repeat\n" + jdump(s['do_not_repeat']))

    raw = ir.get('raw_context') or {}
    for k in ('search_state_note', 'feedback', 'recent_real_evaluated', 'recent_trials'):
        if raw.get(k):
            P.append(f"## {k.replace('_',' ').title()}\n{raw[k]}")
    if raw.get('target_context'):
        P.append("## Target context\n" + jdump(raw['target_context'], indent=1)[:4000])
    if include_parent_artifact and raw.get('parent_train_py'):
        P.append("## Current parent artifact\n" + raw['parent_train_py'])

    req = [f"Allowed actions: {r['allowed_actions']}",
           f"Number of candidates: {r['num_candidates']}"]
    if r.get('max_edits_per_candidate'):
        req.append(f"Max edits per candidate: {r['max_edits_per_candidate']}")
    if r.get('description'): req.append(r['description'])
    req.append('\nReturn a single JSON object: {"type": ..., "reasoning": ..., '
               '"payload": ..., "summary": ...}')
    P.append("## Your move\n" + "\n".join(req))

    return "\n\n".join(P)

def render_record(ir, mode="prose", include_parent_artifact=True):
    if mode == "json":
        shown = {k: v for k, v in ir.items() if k != 'action'}
        instruction = jdump(shown, indent=1)
    else:
        instruction = render_prose(ir, include_parent_artifact)
    return {
        "system": SYSTEM_BY_TASK.get(ir['task']['id'], GENERIC_SYSTEM),
        "instruction": instruction,
        "input": "",
        "output": jdump(ir['action']),
        "source": ir['task']['id'],
        "action_type": ir['action']['type'],
    }

# ----------------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------------
def cmd_from_sample(a):
    data = json.load(open(a.__dict__['in'], encoding='utf-8'))
    out, fail = [], collections.Counter()
    for task, recs in data.items():
        fn = ADAPTERS.get(task)
        if not fn:
            print(f"[warn] no adapter for task '{task}', skipped"); continue
        for r in recs:
            ir = fn(r)
            if ir is None: fail[task] += 1; continue
            out.append(ir)
    write_jsonl(out, a.out_ir)
    print(f"[from-sample] wrote {len(out)} IR records -> {a.out_ir}"
          + (f"  (adapter failures: {dict(fail)})" if fail else ""))
    _audit(out)

def cmd_from_nanogpt_run(a):
    RANK = {"generation_error": 0, "seed": 1, "generated": 2,
            "surrogate_scored": 3, "crash": 4, "evaluated": 5}
    run = Path(a.run_dir)
    status = {}
    mf = run / "manifest.jsonl"
    if mf.exists():
        for l in mf.open(encoding='utf-8'):
            try: d = json.loads(l)
            except Exception: continue
            if d.get('state_id'): status[d['state_id']] = d.get('status')
    floor = RANK.get(a.min_status, 0)
    out, skipped = [], 0
    for sd in sorted((run / "states").glob("state_*")):
        p, rp = sd / "prompt.md", sd / "response.md"
        if not (p.exists() and rp.exists()): skipped += 1; continue
        if RANK.get(status.get(sd.name), 0) < floor: skipped += 1; continue
        od = {}
        opj = sd / "operations.json"
        if opj.exists():
            try:
                od = json.loads(opj.read_text(encoding='utf-8'))
            except Exception:
                od = {}
        ir = adapt_nanogpt({"instruction": p.read_text(encoding='utf-8'),
                            "output": rp.read_text(encoding='utf-8'),
                            "operations_doc": od})
        if ir is None: skipped += 1; continue
        ir['search_state']['status'] = status.get(sd.name)
        out.append(ir)
    write_jsonl(out, a.out_ir)
    print(f"[from-nanogpt-run] wrote {len(out)} IR records (skipped {skipped}) -> {a.out_ir}")
    _audit(out)

def cmd_render(a):
    irs = read_jsonl(a.__dict__['in_ir'])
    recs = [render_record(ir, a.render, not a.strip_parent_artifact) for ir in irs]
    write_jsonl(recs, a.out)
    print(f"[render] {len(recs)} records ({a.render}) -> {a.out}")
    if a.dataset_info:
        info = {"ldm_bo_sft": {"file_name": Path(a.out).name, "formatting": "alpaca",
                               "columns": {"prompt": "instruction", "query": "input",
                                           "response": "output", "system": "system"}}}
        Path(a.dataset_info).write_text(jdump(info, indent=2), encoding='utf-8')
        print(f"[render] dataset_info -> {a.dataset_info}")

def _audit(irs):
    by_task = collections.Counter(ir['task']['id'] for ir in irs)
    by_act = collections.Counter(ir['action']['type'] for ir in irs)
    print("\n  --- IR audit ---")
    print("  by task:  ", dict(by_task))
    print("  by action:", dict(by_act))
    tot = sum(by_act.values()) or 1
    for k, v in by_act.items():
        print(f"    {k:22s} {v:5d}  {100*v/tot:5.2f}%")
    # mode-collapse probe: which parameters get edited
    pc = collections.Counter()
    for ir in irs:
        if ir['action']['type'] != 'propose': continue
        for c in ir['action']['payload'].get('candidates', []):
            for e in (c.get('edits') or []):
                pc[e.get('parameter')] += 1
    if pc:
        t = sum(pc.values())
        print("  edited parameters (top 6):")
        for n, v in pc.most_common(6):
            print(f"    {n:22s} {v:5d}  {100*v/t:5.1f}%")
        top2 = sum(v for _, v in pc.most_common(2))
        print(f"    -> top-2 concentration: {100*top2/t:.1f}%")
    exp = by_act.get('expand_design_space', 0) + by_act.get('add_new_parameter', 0)
    if exp / tot < 0.05:
        print(f"  [!] space-expanding actions are {100*exp/tot:.2f}% of data - "
              f"SFT will likely drop this behaviour. Consider oversampling or "
              f"collecting more expansion traces.")

def cmd_audit(a):
    _audit(read_jsonl(a.__dict__['in_ir']))

def main():
    ap = argparse.ArgumentParser(description="Build ldm-2.0 IR and render to LlamaFactory.")
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('from-sample', help='convert {smallmol,nanogpt,protein} sample json')
    p.add_argument('--in', required=True, dest='in')
    p.add_argument('--out-ir', required=True)
    p.set_defaults(func=cmd_from_sample)

    p = sub.add_parser('from-nanogpt-run', help='convert a full nanogpt run directory')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--out-ir', required=True)
    p.add_argument('--min-status', default='evaluated')
    p.set_defaults(func=cmd_from_nanogpt_run)

    p = sub.add_parser('render', help='ldm-2.0 IR -> LlamaFactory alpaca')
    p.add_argument('--in-ir', required=True, dest='in_ir')
    p.add_argument('--out', required=True)
    p.add_argument('--render', default='prose', choices=['prose', 'json'])
    p.add_argument('--strip-parent-artifact', action='store_true',
                   help='drop the embedded parent train.py (saves ~70%% tokens; '
                        'breaks train/inference prompt parity)')
    p.add_argument('--dataset-info', default=None)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser('audit', help='action-type / mode-collapse audit')
    p.add_argument('--in-ir', required=True, dest='in_ir')
    p.set_defaults(func=cmd_audit)

    a = ap.parse_args()
    a.func(a)

if __name__ == '__main__':
    main()
