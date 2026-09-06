# oldID -> newID lineage

Original agents died ~23:07Z 2026-09-05 (session quota). Their durable
transcripts were searched across every session dir and are **ABSENT** — `/home` is over its
byte quota (105,611,968* / 105,468,748 KB hard), so agent-rescue's layer 3 did not apply.
No transcript and no banked progress survived from any original agent. Roles were recreated
from the bank's role prompt plus, for A3 and A5, preserved partial artifacts.

| role | slug | old ID (dead, unrecoverable) | new ID (running) | inherited partials |
|---|---|---|---|---|
| A2 | provenance | `ae4ee5e962327607a` | `af8bb3aa520e5914f` | none |
| A3 | statistics | `a2625304f8cd8b6ce` | `a0e675a52d822fc08` | a3/ 4 scripts |
| A5 | cost | `aae44e111975f3688` | `a0e14e316e5868a55` | a5/ 8 files, 8.3 MB sacct |
| A1 | main-verdict | `a78e27c0351595e54` | `adf0af1d1ee29b5a0` | none |
| A4 | implementation | `a815988f1836b4b35` | `a1eebb0e9f371adca` | none |

Checkpoint per role: `work/<slug>/progress.md` (append-only), brief `work/<slug>/prompt.txt`.

## 2026-09-06 runset audit (no new roles created)
Performed in the parent session, not by a new agent. No B role was re-dispatched; no
completed A role was reopened. The audit's outputs are `plan/RUNSET_AUDIT.md`,
`results/runset_audit.json` and `plan/runset_audit.py`, mirrored into the private
CLAUDE_CONFIG_DIR at `role-reports/`. Old->new lineage: this continues B1/B3/B4's
duplicate-count dispute using their own sources; their agent ids are unchanged and their
progress files were read, not modified.

## 2026-09-06 B3/B4 resume attempt — REAL FAILURE, work incomplete
Both were dispatched as native agents and both terminated on an API error, not by
completing:

| role | old id | new id | outcome |
|---|---|---|---|
| B4-infra | a47f5020a00fde589 | **a549dfcce58855d90** | FAILED — HTTP 429 session limit (resets 09:40 UTC), req_011Cemjo7CJqzkv9VZkTvLgP, model claude-opus-5. Last line: "Set equality confirmed exactly." |
| B3-metric | a265325be30485b0e | **a6316509970de572b** | FAILED — HTTP 429 session limit, req_011CemjoUEqPCpyWjFahjrER, model claude-opus-5. Last line: "This is decisive. Let me find the upstream SmilesCandidateEvaluator status logic." |

Durable transcripts (the real ones, owner-verified):
`<CLAUDE_CONFIG_DIR>/projects/-lus-lfs1aip2-projects-public-u6gb/3edc5462-19e3-492e-9aeb-534e7b0e31c5/subagents/agent-<id>.jsonl`
— 186,186 B and 147,954 B respectively when checked, each with a `.meta.json`.
The `/run/user/.../tasks/<id>.output` files are 199-byte scratch stubs and are NOT the
durable children; an earlier reply mislabelled them.

Neither agent posted its mutual reply to disk before dying. **B3/B4 mutual replies,
B2/B5 cross-attacks and the Max adjudication are all still OUTSTANDING.** They must be
resumed after the quota reset, continuing only the unfinished part.
