#!/usr/bin/env python3
"""
fabricate.py -- Rule-grounded data augmentation for ldm-2.0 IR (nanogpt-style
parameter_edits tasks).

WHY THIS EXISTS
---------------
The real traces are mode-collapsed: HEAD_DIM + WINDOW_PATTERN account for ~79% of
all edits, and `expand_design_space` is only ~0.5% of actions. A model distilled
from that data locks onto two knobs and never learns to expand the design space.

But the concentration is NOT the disease: those two knobs really are high-leverage
(they drive ~86% of the real improvements). The disease is that the teacher keeps
hammering them *after they stop paying off*. So we do NOT flatten the marginal --
we fabricate the CONDITIONAL behaviour that is missing:
    "when the space is exhausted and progress has stalled, switch or expand."

Every fabricated record is derived from real data + a stated BO rule, never from
invented value judgements. All synthetic records carry `provenance`.

OPERATORS
---------
  F1 exhaustion   : inject evidence that the 2 dominant choice knobs are exhausted
                    (grid enumerated, scores flat within the real noise floor)
                    -> target action = expand_design_space
  F2 plateau      : keep a REAL context whose action repeats an already-tried
                    (param,value); relabel the action per BO rules
  F3 rotation     : mask the dominant knobs out of the active set; transplant a
                    real action that uses a rare parameter
  F4 jitter       : resample numeric values inside their domain (log-aware),
                    keeping the edit's directional rationale
  F5 transplant   : move real expand_design_space actions onto other stalled contexts

CALIBRATION (measured from expanded_ldm_bon_N4H4_03, do not guess these):
  * real val_bpb: min 0.984639 / median 0.989709 / max 1.069486
  * noise floor : median adjacent |dval_bpb| = 2.25e-4  (paper's restored-anchor
                  repeats estimate 3-5e-4) -> DEFAULT_NOISE = 2e-4
  * HEAD_DIM has 3 values, WINDOW_PATTERN has 6 -> only 18 combos exist, and all
    were already tried; the teacher still proposed them ~1200 times each.

USAGE
-----
  python fabricate.py --in-ir ir_ng.jsonl --out-ir ir_ng_aug.jsonl \
      --f1 150 --f2 all --f3 80 --f4 300 --f5 40 --seed 0

  python fabricate.py --in-ir ir_ng.jsonl --out-ir /dev/null --report-only
"""
import argparse, json, math, random, collections, copy
from pathlib import Path

DEFAULT_NOISE = 2e-4          # real noise floor; flat-within-this == "no effect"
DEFAULT_BEST = 0.984639       # real best val_bpb in the reference run
SCHEMA_VERSION = "ldm-2.0"

# ---------------------------------------------------------------- io helpers
def read_jsonl(p):
    t = Path(p).read_text(encoding='utf-8')
    if t.lstrip().startswith('['):
        return json.loads(t)
    return [json.loads(l) for l in t.splitlines() if l.strip()]

def write_jsonl(recs, p):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

# ---------------------------------------------------------------- IR helpers
def is_edit_task(ir):
    return ir['search_state']['design_space'].get('representation') == 'parameter_edits'

def edits_of(ir):
    if ir['action']['type'] != 'propose':
        return []
    out = []
    for c in ir['action']['payload'].get('candidates', []):
        out.extend(c.get('edits') or [])
    return out

def signature(ir):
    return tuple(sorted((e.get('parameter'), str(e.get('value'))) for e in edits_of(ir)))

def active_map(ir):
    return {p['name']: p for p in ir['search_state']['design_space']['active_parameters']}

def stamp(ir, operator, derived_from, rule, extra=None):
    ir['provenance'] = {"synthetic": True, "operator": operator,
                        "derived_from": derived_from, "rule": rule}
    if extra:
        ir['provenance'].update(extra)
    return ir

def rec_id(ir, i):
    return ir['search_state'].get('state_id') or f"state_{ir['search_state'].get('round') or i:04d}"

def sample_value(p, rng, avoid_current=True):
    """Draw a plausible value inside a parameter's own domain.

    Never returns the parameter's CURRENT value: proposing `set HEAD_DIM 128` when
    HEAD_DIM is already 128 is a no-op the runner would apply to no effect. Real
    traces contain 0% such edits, so emitting them would be a synthetic-only artifact.
    """
    d = p.get('domain')
    cur = p.get('current_value')
    if p.get('type') == 'choice':
        opts = [v for v in d if not (avoid_current and cur is not None and v == cur)]
        return rng.choice(opts or d)
    lo, hi = d[0], d[1]
    for _ in range(8):
        if p.get('scale') == 'log' and lo > 0:
            v = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        else:
            v = rng.uniform(lo, hi)
        v = int(round(v)) if p.get('type') == 'int' else round(v, 6)
        if not (avoid_current and cur is not None and v == cur):
            return v
    return v

def edit_op_for(p):
    return p.get('edit_op') or ('set_choice' if p.get('type') == 'choice' else 'set_numeric')

# ---------------------------------------------------------------- corpus stats
def corpus_stats(irs):
    tried = collections.defaultdict(set)
    proposed = collections.Counter()
    sig_count = collections.Counter()
    for ir in irs:
        if not is_edit_task(ir):
            continue
        for e in edits_of(ir):
            tried[e['parameter']].add(str(e.get('value')))
            proposed[e['parameter']] += 1
        s = signature(ir)
        if s:
            sig_count[s] += 1
    return tried, proposed, sig_count

def dominant_params(proposed, k=2):
    return [n for n, _ in proposed.most_common(k)]

def exhausted_choice_params(irs, tried):
    """Choice params whose entire domain has already been tried."""
    out = {}
    for ir in irs:
        for p in ir['search_state']['design_space']['active_parameters']:
            if p.get('type') != 'choice':
                continue
            dom = {str(v) for v in p['domain']}
            if dom and dom.issubset(tried.get(p['name'], set())):
                out[p['name']] = p['domain']
    return out

# ---------------------------------------------------------------- F1 exhaustion
def f1_exhaustion(irs, n, rng, noise, best_val, stall_rounds=15):
    """Inject a context proving the dominant choice knobs are exhausted and flat,
    then require expand_design_space. This is the operator that actually teaches
    'stalled + exhausted -> expand'."""
    donors = [ir for ir in irs
              if is_edit_task(ir) and ir['search_state']['design_space']['inactive_parameters']]
    if not donors:
        return []
    tried, proposed, _ = corpus_stats(irs)
    ex = exhausted_choice_params(irs, tried)
    # prefer the two most-proposed exhausted choice knobs
    grid_names = [n_ for n_ in dominant_params(proposed, 4) if n_ in ex][:2]
    if len(grid_names) < 2:
        return []
    out = []
    for i in range(n):
        src = copy.deepcopy(rng.choice(donors))
        sid = rec_id(src, i)
        doms = [ex[g] for g in grid_names]
        combos = [(a, b) for a in doms[0] for b in doms[1]]

        # The *spread* of the synthetic grid must land within the noise floor -- that
        # is the evidence the reasoning will cite. For k draws from N(0, sigma) the
        # expected range is ~3.5-4 sigma, so sigma must be noise/4, NOT noise.
        # (Getting this wrong makes the record assert a flatness its own numbers
        # contradict, which would train the model to make false claims.)
        sigma = noise / 4.0
        obs, dnr = [], []
        for a, b in combos:
            score = round(best_val + rng.gauss(0, sigma), 6)
            obs.append({
                "design": {grid_names[0]: a, grid_names[1]: b},
                "results": {"val_bpb": score},
                "roles": ["recent"],
                "description": f"{grid_names[0]}={a}, {grid_names[1]}={b}",
            })
            dnr.append({grid_names[0]: a, grid_names[1]: b})
        spread = max(o['results']['val_bpb'] for o in obs) - min(o['results']['val_bpb'] for o in obs)
        # state the evidence truthfully, whatever the draw produced
        if spread <= noise:
            claim = f"range {spread:.6f}, not above the noise floor (~{noise:.0e})"
        else:
            claim = f"range {spread:.6f} (about {spread/noise:.1f}x the noise floor {noise:.0e})"

        ss = src['search_state']
        ss['observations'] = obs
        ss['num_evaluated'] = len(obs)
        # deliberately NOT setting do_not_repeat: nanogpt has no such constraint in
        # the real prompts, so populating it only here would (a) be a fabricated
        # instruction and (b) make the field's presence a perfect predictor of
        # expand_design_space. The exhaustion evidence is in observations+progress.
        ss['do_not_repeat'] = []
        ss['best_so_far'] = min(obs, key=lambda o: o['results']['val_bpb'])
        ss['progress'] = {
            "stalled": True, "rounds_since_improvement": stall_rounds,
            "description": (f"All {len(combos)} combinations of {grid_names[0]} x {grid_names[1]} "
                            f"have been evaluated; {claim}; no improvement for "
                            f"{stall_rounds} consecutive rounds."),
        }
        # CRITICAL: the donor's real prose (feedback / recent trials) describes a
        # DIFFERENT history and would contradict the synthetic evidence inside the
        # same prompt -- the reasoning would cite an exhaustion the prompt itself
        # disproves, training the model to invent justifications. Rewrite the prose
        # so it agrees with the fabricated observations.
        rc = src.setdefault('raw_context', {})
        best = ss['best_so_far']
        rc['feedback'] = (
            f"Current best real evaluation: val_bpb={best['results']['val_bpb']:.6f} "
            f"status=evaluated action=set {grid_names[0]} "
            f"{best['design'][grid_names[0]]}; set {grid_names[1]} {best['design'][grid_names[1]]}\n"
            f"No improvement in the last {stall_rounds} rounds.")
        rc['recent_real_evaluated'] = "\n".join(
            f"- {grid_names[0]}={o['design'][grid_names[0]]}, "
            f"{grid_names[1]}={o['design'][grid_names[1]]}: "
            f"val_bpb={o['results']['val_bpb']:.6f} (lower is better); status=evaluated"
            for o in obs)
        rc['recent_trials'] = "\n".join(
            f"- iter={k} status=evaluated val_bpb={o['results']['val_bpb']:.6f} (valid); "
            f"action=set {grid_names[0]} {o['design'][grid_names[0]]}; "
            f"set {grid_names[1]} {o['design'][grid_names[1]]}"
            for k, o in enumerate(obs))
        rc['search_state_note'] = (
            f"- Parent depth: 0\n- Parent val_bpb: {best['results']['val_bpb']:.6f}\n"
            f"- Search note: {grid_names[0]} x {grid_names[1]} grid exhausted "
            f"({len(combos)} combos); stalled for {stall_rounds} rounds.")

        inact = ss['design_space']['inactive_parameters']
        pick = rng.choice(inact)
        src['request']['allowed_actions'] = ["propose", "expand_design_space"]
        src['action'] = {
            "type": "expand_design_space",
            "reasoning": (f"The value domains of {grid_names[0]} and {grid_names[1]} have been exhausted "
                          f"({len(combos)} combinations evaluated); {claim}. This indicates that the "
                          f"current active feature set no longer explains the remaining variation. "
                          f"Continuing to edit inside this subspace has no information gain, so "
                          f"activate {pick['name']} to expand the surrogate feature space."),
            "payload": {"activate": pick['name'],
                        "initial_value": pick.get('current_value')},
            "summary": f"activate {pick['name']} (exhausted {'+'.join(grid_names)} grid, stalled {stall_rounds} rounds)",
        }
        # the grid knobs are exhausted -> they are no longer useful targets
        out.append(stamp(src, "F1_exhaustion", sid,
                         f"{'x'.join(grid_names)} grid exhausted ({len(combos)} combos), "
                         f"spread {spread:.6f} vs noise floor {noise:.0e} -> expand",
                         {"noise_scale": noise, "sigma": sigma, "spread": round(spread, 6),
                          "grid": grid_names,
                          "synthetic_fields": ["observations", "do_not_repeat",
                                               "best_so_far", "progress", "action",
                                               "raw_context.feedback",
                                               "raw_context.recent_real_evaluated",
                                               "raw_context.recent_trials",
                                               "raw_context.search_state_note"]}))
    return out

# ---------------------------------------------------------------- F2 plateau relabel
def f2_plateau_relabel(irs, rng, min_repeat, limit=None):
    """REAL context, fabricated action. Targets records that repeat a (param,value)
    signature the corpus has already tried many times -- i.e. the exact 'hammering a
    dead knob' pathology."""
    tried, proposed, sig_count = corpus_stats(irs)
    cands = [ir for ir in irs
             if is_edit_task(ir) and ir['action']['type'] == 'propose'
             and signature(ir) and sig_count[signature(ir)] >= min_repeat]

    # Prioritise records whose own context says the search has STALLED. Those are
    # precisely the ones teaching "stalled -> keep hammering the same knob": with the
    # `progress` field now visible in the prompt, leaving them unrelabelled would make
    # that pathology explicit and cancel out what F1 is trying to teach.
    def stalled(ir):
        pg = ir['search_state'].get('progress') or {}
        return bool(pg.get('stalled'))
    rng.shuffle(cands)
    cands.sort(key=lambda ir: not stalled(ir))   # stalled first, order otherwise random
    n_stalled = sum(1 for c in cands if stalled(c))
    if limit:
        cands = cands[:limit]
    kept_stalled = sum(1 for c in cands if stalled(c))
    print(f"    [F2] {n_stalled} stalled candidates available; "
          f"relabelling {kept_stalled} of them (+{len(cands)-kept_stalled} non-stalled)")
    out = []
    for i, src0 in enumerate(cands):
        src = copy.deepcopy(src0)
        sid = rec_id(src, i)
        sig = signature(src0)
        reps = sig_count[sig]
        amap = active_map(src)
        inact = src['search_state']['design_space']['inactive_parameters']
        old = "; ".join(f"{p}={v}" for p, v in sig)

        # Rule 1: an inactive dimension exists -> expand
        if inact:
            pick = rng.choice(inact)
            src['request']['allowed_actions'] = ["propose", "expand_design_space"]
            src['action'] = {
                "type": "expand_design_space",
                "reasoning": (f"This edit ({old}) has been proposed {reps} times in history without "
                              f"improvement, so repeating it has no information gain. The current "
                              f"active feature set has been sampled enough; activate {pick['name']} "
                              f"instead to expand the feature space."),
                "payload": {"activate": pick['name'], "initial_value": pick.get('current_value')},
                "summary": f"activate {pick['name']} instead of repeating {old}",
            }
            rule = f"signature repeated {reps}x with no improvement -> expand"
        else:
            # Rule 2: prefer a never-tried active param, else the least-tried one
            untried = [n for n in amap if not tried.get(n)]
            pool = untried or sorted(amap, key=lambda n: proposed.get(n, 0))
            pname = pool[0] if untried else rng.choice(pool[:3])
            p = amap[pname]
            v = sample_value(p, rng)
            ntried = len(tried.get(pname, []))
            ndom = len(p['domain']) if p.get('type') == 'choice' else 'continuous'
            src['action'] = {
                "type": "propose",
                "reasoning": (f"This edit ({old}) has been proposed {reps} times in history without "
                              f"improvement; repeating it has no information gain. Switch to the "
                              f"under-sampled parameter {pname} (tried {ntried} value(s) / domain {ndom})."),
                "payload": {"candidates": [{"parent": None, "edits": [
                    {"parameter": pname, "edit_op": edit_op_for(p), "value": v,
                     "rationale": f"{pname} is under-sampled (tried {ntried} value(s)); explore uncovered regions"}]}]},
                "summary": f"switch to under-sampled {pname} instead of repeating {old}",
            }
            rule = f"signature repeated {reps}x -> switch to under-sampled {pname}"
        out.append(stamp(src, "F2_plateau_relabel", sid, rule,
                         {"repeat_count": reps, "original_action": sig,
                          "synthetic_fields": ["action"]}))
    return out

# ---------------------------------------------------------------- F3 knob rotation
def f3_rotation(irs, n, rng):
    """Counterfactual: mask the dominant knobs out of the active set, then transplant
    a REAL action that uses a rare parameter. Teaches 'what else is available'.
    Keep the share low -- the model must not conclude the main knobs are usually gone."""
    tried, proposed, _ = corpus_stats(irs)
    dom = set(dominant_params(proposed, 2))
    donors = [ir for ir in irs
              if is_edit_task(ir) and ir['action']['type'] == 'propose'
              and edits_of(ir) and not (set(e['parameter'] for e in edits_of(ir)) & dom)]
    hosts = [ir for ir in irs
             if is_edit_task(ir) and ir['action']['type'] == 'propose'
             and (set(e['parameter'] for e in edits_of(ir)) & dom)]
    if not donors or not hosts:
        return []
    out = []
    for i in range(min(n, len(hosts))):
        src = copy.deepcopy(rng.choice(hosts))
        sid = rec_id(src, i)
        ds = src['search_state']['design_space']
        kept = [p for p in ds['active_parameters'] if p['name'] not in dom]
        if not kept:
            continue
        ds['active_parameters'] = kept
        ds['description'] = ((ds.get('description') or '') +
                             f"\nNote: the value domains of {', '.join(sorted(dom))} are exhausted "
                             f"and frozen, so they cannot be edited this round.")
        # CRITICAL: the donor's real prose still shows the masked knobs being edited,
        # which contradicts the "frozen" claim inside the same prompt. Drop the
        # trial-level prose rather than fabricate a fake history for it; the
        # structured history stays authoritative.
        rc = src.setdefault('raw_context', {})
        for k in ('feedback', 'recent_real_evaluated', 'recent_trials'):
            rc.pop(k, None)
        rc['search_state_note'] = (
            f"- Search note: {', '.join(sorted(dom))} frozen (domain exhausted); "
            f"edit only the remaining active parameters.")
        donor = rng.choice(donors)
        act = copy.deepcopy(donor['action'])
        # keep only edits whose parameter survives the mask AND that actually change
        # something: the donor's value may coincide with THIS host's current value,
        # producing a no-op the real traces never contain.
        names = {p['name'] for p in kept}
        cmap = {p['name']: p.get('current_value') for p in kept}
        for c in act['payload'].get('candidates', []):
            c['edits'] = [e for e in (c.get('edits') or [])
                          if e['parameter'] in names
                          and not (cmap.get(e['parameter']) is not None
                                   and e.get('value') == cmap[e['parameter']])]
        if not any(c.get('edits') for c in act['payload'].get('candidates', [])):
            continue
        used = sorted({e['parameter'] for c in act['payload']['candidates'] for e in c['edits']})
        act['reasoning'] = (f"{', '.join(sorted(dom))} are frozen because their value domains are exhausted; "
                            f"edit the remaining active parameter(s): {', '.join(used)}.")
        src['action'] = act
        out.append(stamp(src, "F3_rotation", sid,
                         f"masked {sorted(dom)} from active set; transplanted real action on {used}",
                         {"masked": sorted(dom), "donor": rec_id(donor, -1),
                          "synthetic_fields": ["design_space.active_parameters", "action",
                                               "raw_context(trial prose removed)"]}))
    return out

# ---------------------------------------------------------------- F4 value jitter
def f4_jitter(irs, n, rng):
    """Resample numeric values inside their own domain. Only the value moves; the
    decision logic and the directional rationale are untouched. Cheapest, safest
    coverage gain for the numeric knobs (SCALAR_LR has just 1 real sample)."""
    hosts = [ir for ir in irs
             if is_edit_task(ir) and ir['action']['type'] == 'propose'
             and any(e.get('edit_op') == 'set_numeric' for e in edits_of(ir))]
    if not hosts:
        return []
    out = []
    for i in range(n):
        src = copy.deepcopy(rng.choice(hosts))
        sid = rec_id(src, i)
        amap = active_map(src)
        changed = []
        for c in src['action']['payload'].get('candidates', []):
            for e in (c.get('edits') or []):
                p = amap.get(e['parameter'])
                if not p or e.get('edit_op') != 'set_numeric':
                    continue
                old = e['value']
                e['value'] = sample_value(p, rng)   # never equals current_value
                changed.append(f"{e['parameter']}:{old}->{e['value']}")
        if not changed:
            continue
        out.append(stamp(src, "F4_jitter", sid,
                         "numeric values resampled within their declared domain "
                         "(log-uniform where scale=log); decision logic unchanged",
                         {"changes": changed, "synthetic_fields": ["action.payload.value"]}))
    return out

# ---------------------------------------------------------------- F5 transplant
def f5_transplant(irs, n, rng):
    """Move REAL expand_design_space actions onto other contexts that have the same
    inactive dimension available. Teaches the action's association with the stall
    signal rather than with one specific context."""
    real_exp = [ir for ir in irs if ir['action']['type'] == 'expand_design_space']
    if not real_exp:
        return []
    out = []
    for i in range(n):
        donor = rng.choice(real_exp)
        target = donor['action']['payload'].get('activate')
        hosts = [ir for ir in irs
                 if is_edit_task(ir) and ir['action']['type'] == 'propose'
                 and any(p['name'] == target
                         for p in ir['search_state']['design_space']['inactive_parameters'])]
        if not hosts:
            continue
        src = copy.deepcopy(rng.choice(hosts))
        sid = rec_id(src, i)
        src['request']['allowed_actions'] = ["propose", "expand_design_space"]
        src['action'] = copy.deepcopy(donor['action'])
        if not src['action'].get('reasoning'):
            src['action']['reasoning'] = (f"The current active feature set does not explain the remaining "
                                          f"variation well enough; activate {target} to expand the feature space.")
        out.append(stamp(src, "F5_transplant", sid,
                         f"real expand action (activate {target}) transplanted onto a "
                         f"context exposing the same inactive dimension",
                         {"donor_activate": target, "synthetic_fields": ["action"]}))
    return out

# ---------------------------------------------------------------- report
def _validity_check(irs):
    """Every action must be executable by the runner: parameters active, values in
    domain, edit budget respected, activate target genuinely inactive.

    This once caught training targets taken from REJECTED transcript attempts
    (out-of-range values, 3 edits where the budget is 2) -- i.e. teaching the model
    to emit actions the runner throws away.
    """
    fails = collections.Counter()
    for ir in irs:
        ds = ir['search_state']['design_space']
        act = ir['action']
        amap = {p['name']: p for p in ds['active_parameters']}
        imap = {p['name']: p for p in ds['inactive_parameters']}
        if act['type'] not in (ir['request'].get('allowed_actions') or []):
            fails['action_type_not_allowed'] += 1
        maxe = ir['request'].get('max_edits_per_candidate')
        if act['type'] == 'propose':
            for c in act['payload'].get('candidates', []):
                edits = c.get('edits') or []
                if maxe and len(edits) > maxe:
                    fails['too_many_edits'] += 1
                for e in edits:
                    p = amap.get(e.get('parameter'))
                    if p is None:
                        fails['param_not_active'] += 1
                        continue
                    v = e.get('value')
                    if p.get('type') == 'choice':
                        if v not in p['domain']:
                            fails['value_not_in_choice'] += 1
                    else:
                        try:
                            if not (p['domain'][0] <= v <= p['domain'][1]):
                                fails['value_out_of_range'] += 1
                        except TypeError:
                            fails['value_type_bad'] += 1
                        if p.get('type') == 'int' and not isinstance(v, int):
                            fails['int_param_got_float'] += 1
                    # a no-op edit (value == current) is not a proposal; real traces
                    # contain 0% of these, so any is a synthetic artifact
                    if p.get('current_value') is not None and v == p['current_value']:
                        fails['noop_edit'] += 1
        elif act['type'] == 'expand_design_space':
            if act['payload'].get('activate') not in imap:
                fails['activate_not_inactive'] += 1
    if fails:
        print("  [FAIL] invalid actions (the runner would reject these):")
        for k, v in fails.most_common():
            print(f"    {k:28s} {v}")
    else:
        print("  [ok] every action is valid against its own declared design space")
    return not fails

def _consistency_check(irs):
    """Two prompt-level pathologies that silently poison training."""
    ok = True

    # (a) The prompt states a hard exclusion the target action then violates.
    #     (Once true for 1471/1534 records after a fabricated `do_not_repeat`.)
    viol = 0
    for ir in irs:
        dnr = ir['search_state'].get('do_not_repeat') or []
        if not dnr or ir['action']['type'] != 'propose':
            continue
        excluded = set()
        for d in dnr:
            excluded.add(json.dumps(d, sort_keys=True, ensure_ascii=False)
                         if isinstance(d, (dict, list)) else str(d))
        for c in ir['action']['payload'].get('candidates', []):
            d = c.get('design')
            key = (json.dumps(d, sort_keys=True, ensure_ascii=False)
                   if isinstance(d, (dict, list)) else str(d))
            if key in excluded:
                viol += 1
                break
    if viol:
        ok = False
        print(f"  [FAIL] {viol} records propose a design their own prompt lists under "
              f"`do_not_repeat` -- this trains the model to disobey instructions")
    else:
        print("  [ok] no record violates its own do_not_repeat constraint")

    # (b) `progress.stalled` is visible in the prompt, so records that stall and then
    #     hammer on regardless actively teach the pathology we are trying to remove.
    tab = collections.Counter()
    for ir in irs:
        pg = ir['search_state'].get('progress') or {}
        if pg.get('stalled'):
            tab[ir['action']['type']] += 1
    tot = sum(tab.values())
    if tot:
        pr = tab.get('propose', 0)
        share = 100 * pr / tot
        if share > 85:
            print(f"  [warn] of {tot} stalled records, {share:.1f}% still `propose` -- "
                  f"the prompt shows the stall, so these teach 'stalled -> keep going'. "
                  f"Raise --f2 to relabel more of them.")
        else:
            print(f"  [ok] stalled records: {share:.1f}% propose / "
                  f"{100-share:.1f}% expand -- stall signal is actionable")
    return ok

def _shortcut_check(irs):
    """Detect structural giveaways: if a field is present on synthetic records but
    absent on real ones, the rendered heading alone predicts the action and the model
    learns the marker instead of the reasoning. This exact leak once made every
    '## Observed history' record an expand_design_space record."""
    probes = {
        'observations': lambda ir: bool(ir['search_state'].get('observations')),
        'progress': lambda ir: ir['search_state'].get('progress') is not None,
        'surrogate_feedback': lambda ir: ir['search_state'].get('surrogate_feedback') is not None,
        'do_not_repeat': lambda ir: bool(ir['search_state'].get('do_not_repeat')),
    }
    leaks = []
    for name, fn in probes.items():
        tab = collections.Counter((fn(ir), ir['action']['type']) for ir in irs)
        present = {a: n for (p, a), n in tab.items() if p}
        if len(present) == 1 and sum(present.values()) > 10:
            only = next(iter(present))
            absent = {a for (p, a), n in tab.items() if not p}
            if absent - {only}:
                leaks.append((name, only, sum(present.values())))
    if leaks:
        print("  [FAIL] structural shortcut(s) -- field presence perfectly predicts the action:")
        for n, a, c in leaks:
            print(f"    '{n}' present on {c} records, ALL of them action='{a}'")
    else:
        print("  [ok] no structural shortcut between field presence and action type")
    return not leaks

def verify(irs):
    """Guard against fabricating claims the record's own prompt contradicts.
    A synthetic record whose reasoning cites evidence its own context disproves
    would train the model to invent justifications -- worse than not augmenting."""
    ok = True

    # (1) F1: the asserted flatness must match the synthetic numbers.
    bad = []
    for ir in irs:
        pv = ir.get('provenance') or {}
        if pv.get('operator') != 'F1_exhaustion':
            continue
        vals = [o['results']['val_bpb'] for o in ir['search_state']['observations']]
        spread = max(vals) - min(vals)
        noise = pv.get('noise_scale')
        txt = (ir['action'].get('reasoning') or '') + (
            ir['search_state'].get('progress', {}).get('description') or '')
        if ('not above the noise floor' in txt or 'within the noise floor' in txt) and noise and spread > noise * 1.0001:
            bad.append((pv.get('derived_from'), spread, noise))
    if bad:
        ok = False
        print(f"  [FAIL] {len(bad)} F1 records claim flatness their numbers contradict:")
        for d, s, n in bad[:3]:
            print(f"    {d}: spread {s:.6f} > noise floor {n:.0e}")
    else:
        print("  [ok] F1 flatness claims are consistent with the synthetic numbers")

    # (2) F1: retained prose must not describe a different history than the
    #     synthetic observations (this once put two contradictory histories in
    #     the same prompt).
    bad = []
    for ir in irs:
        pv = ir.get('provenance') or {}
        if pv.get('operator') != 'F1_exhaustion':
            continue
        vals = {round(o['results']['val_bpb'], 6) for o in ir['search_state']['observations']}
        prose = " ".join(str(ir.get('raw_context', {}).get(k, ''))
                         for k in ('feedback', 'recent_real_evaluated', 'recent_trials'))
        import re as _re
        quoted = {round(float(x), 6) for x in _re.findall(r'val_bpb=([\d.]+)', prose)}
        alien = {v for v in quoted if v not in vals}
        if alien:
            bad.append((pv.get('derived_from'), sorted(alien)[:3]))
    if bad:
        ok = False
        print(f"  [FAIL] {len(bad)} F1 records retain real prose contradicting their "
              f"synthetic history:")
        for d, a in bad[:3]:
            print(f"    {d}: prose cites val_bpb {a} absent from synthetic observations")
    else:
        print("  [ok] F1 prose agrees with the synthetic history")

    # (3) F3: knobs claimed frozen must not appear as edited in retained prose.
    bad = []
    for ir in irs:
        pv = ir.get('provenance') or {}
        if pv.get('operator') != 'F3_rotation':
            continue
        prose = " ".join(str(ir.get('raw_context', {}).get(k, ''))
                         for k in ('feedback', 'recent_real_evaluated', 'recent_trials'))
        hit = [m for m in pv.get('masked', []) if m in prose]
        if hit:
            bad.append((pv.get('derived_from'), hit))
    if bad:
        ok = False
        print(f"  [FAIL] {len(bad)} F3 records show 'frozen' knobs still being edited "
              f"in their prose:")
        for d, h in bad[:3]:
            print(f"    {d}: {h}")
    else:
        print("  [ok] F3 frozen-knob claims are not contradicted by retained prose")

    # (4) duplicate (context, action) pairs inflate loss on a few templates.
    import hashlib
    seen = collections.Counter()
    for ir in irs:
        key = hashlib.sha256(
            (json.dumps(ir['search_state'], sort_keys=True, ensure_ascii=False)
             + json.dumps(ir['action'], sort_keys=True, ensure_ascii=False)
             ).encode()).hexdigest()
        seen[key] += 1
    dups = sum(v - 1 for v in seen.values() if v > 1)
    if dups:
        print(f"  [warn] {dups} duplicate (context,action) synthetic records "
              f"({100*dups/max(len(irs),1):.1f}%) -- they will be over-weighted in the loss")
    else:
        print("  [ok] no duplicate synthetic (context,action) pairs")
    return ok

def report(irs, title):
    tot = len(irs) or 1
    acts = collections.Counter(ir['action']['type'] for ir in irs)
    syn = sum(1 for ir in irs if ir.get('provenance', {}).get('synthetic'))
    pc = collections.Counter()
    for ir in irs:
        for e in edits_of(ir):
            pc[e['parameter']] += 1
    print(f"\n===== {title} =====")
    print(f"  records: {tot}  (synthetic {syn}, {100*syn/tot:.1f}%)")
    print("  action types:")
    for k, v in acts.most_common():
        print(f"    {k:22s} {v:5d}  {100*v/tot:5.2f}%")
    if pc:
        t = sum(pc.values())
        print("  edited parameters:")
        for k, v in pc.most_common(8):
            print(f"    {k:22s} {v:5d}  {100*v/t:5.1f}%")
        top2 = sum(v for _, v in pc.most_common(2))
        print(f"    -> top-2 concentration: {100*top2/t:.1f}%")
        print(f"    -> parameters ever touched: {len(pc)}")
    by_op = collections.Counter(ir.get('provenance', {}).get('operator', 'REAL') for ir in irs)
    print("  by provenance:", dict(by_op))

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Rule-grounded fabrication for ldm-2.0 IR.")
    ap.add_argument('--in-ir', required=True, dest='in_ir')
    ap.add_argument('--out-ir', required=True)
    ap.add_argument('--f1', default='150', help='exhaustion injections (int)')
    ap.add_argument('--f2', default='all', help='plateau relabels: int or "all"')
    ap.add_argument('--f3', default='80', help='knob rotations (int)')
    ap.add_argument('--f4', default='300', help='value jitters (int)')
    ap.add_argument('--f5', default='40', help='action transplants (int)')
    ap.add_argument('--f2-min-repeat', type=int, default=5,
                    help='minimum times a (param,value) signature must already appear '
                         'before its record is treated as a plateau repeat')
    ap.add_argument('--noise', type=float, default=DEFAULT_NOISE,
                    help='val_bpb noise floor used to make synthetic scores "flat" '
                         '(measured 2.25e-4 in the reference run)')
    ap.add_argument('--best-val', type=float, default=DEFAULT_BEST)
    ap.add_argument('--stall-rounds', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--report-only', action='store_true')
    a = ap.parse_args()

    rng = random.Random(a.seed)
    irs = read_jsonl(a.in_ir)
    real = [ir for ir in irs if not ir.get('provenance', {}).get('synthetic')]
    report(real, 'BEFORE (real data only)')

    n_edit = sum(1 for ir in real if is_edit_task(ir))
    if n_edit == 0:
        print("\n[!] no parameter_edits records found; these operators only apply to "
              "edit-style tasks (e.g. nanogpt). Nothing to do.")
        return

    def cnt(v, default):
        if v == 'all':
            return None
        try:
            return int(v)
        except ValueError:
            return default

    made = []
    made += f1_exhaustion(real, cnt(a.f1, 150) or 150, rng, a.noise, a.best_val, a.stall_rounds)
    made += f2_plateau_relabel(real, rng, a.f2_min_repeat, cnt(a.f2, None))
    made += f3_rotation(real, cnt(a.f3, 80) or 80, rng)
    made += f4_jitter(real, cnt(a.f4, 300) or 300, rng)
    made += f5_transplant(real, cnt(a.f5, 40) or 40, rng)

    print(f"\n  fabricated: {len(made)} records")
    for k, v in collections.Counter(m['provenance']['operator'] for m in made).most_common():
        print(f"    {k:22s} {v:5d}")
    verify(made)

    out = real + made
    _validity_check(out)
    _shortcut_check(out)
    _consistency_check(out)
    report(out, 'AFTER (real + synthetic)')

    print("\n  reminders:")
    print("   * every synthetic record carries `provenance.synthetic = true`")
    print("   * EVALUATE ON REAL HELD-OUT DATA ONLY -- never on synthetic records")
    print("   * top-2 concentration is meant to DROP, not to reach uniform: the two "
          "dominant knobs really do drive ~86% of real improvements")
    print("   * disclose the operators and mix ratio if this feeds a paper")

    if not a.report_only:
        write_jsonl(out, a.out_ir)
        print(f"\n  wrote {len(out)} records -> {a.out_ir}")

if __name__ == '__main__':
    main()
