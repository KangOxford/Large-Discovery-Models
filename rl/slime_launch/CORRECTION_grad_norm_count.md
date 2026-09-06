# Correction: the "238/238 non-finite" claim is wrong

Source: the research supervisor's own independent read of all seven historical 9B
`train.log` files, 2026-09-06 00:24 UTC. Recorded here verbatim as the authoritative count,
superseding the audit's figure.

| | count |
|---|---:|
| `train/grad_norm` records across the seven 9B runs | **238** |
| **nan** | **209** |
| **inf** | **0** |
| **finite** | **29** |

Per run: R2's three runs hold **8 / 14 / 7** finite norms. **R1 logs no norm records at all.**

## What must be corrected

`plan/MAIN_OBJECTIVE_AUDIT.md` and `results/main_objective_manifest.json` both publish
"238/238 grad_norm nan/inf" and "no 9B run has completed an optimizer step with a finite
gradient". **The count is false and must be replaced with 209 nan / 0 inf / 29 finite.**

The all-`inf`-zero detail also matters on its own: the audit wrote "nan/inf" as if both
occurred. Zero infinities means no overflow was ever recorded, which is a different failure
signature from what the audit implied.

## What this does NOT establish — stated because the temptation runs the other way

29 finite gradient norms do **not** show healthy training, and do **not** show any RL
benefit. Specifically:

- A finite logged norm is a **logging-path** observation. It does not by itself show that
  an optimizer step executed, that the update was applied, or that the resulting weights
  are finite.
- Magnitude alone cannot classify a norm as healthy. Whether it was scaled or unscaled,
  pre-clip or post-clip, and in what units, all change the reading. See the open question
  put to A3.
- **It does not change the GATE result.** `6324130.1593` produced **zero** verified finite
  training output — 0 grad_norm records of any kind, 0 loss records, 0 `found_inf`, no
  optimizer update and no weight fingerprint. That remains "not reached".

## Consequence for the audit's blocking-gap claim

The audit's stated blocking gap ("no 9B run ever completed a finite-gradient optimizer
step") rested on the 238/238 figure. With 29 finite norms on record the *evidence string*
collapses. Whether the underlying claim survives is a separate question now under
adversarial review (A1 argues it is refuted via `exp_avg`; that argument is itself under
challenge — see the open question put to A4).

**Count correction and scientific conclusion are kept separate here deliberately.** The
count is settled. The conclusion is not.
