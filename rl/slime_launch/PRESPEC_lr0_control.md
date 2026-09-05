# Pre-specification: additional lr=0 control runs

Frozen 2026-09-05 06:00Z, before the second control run started.

## Scope, stated first because it limits everything below

**This is prospective for the four ADDITIONAL control runs only. It is not a
pre-specification of the comparison as a whole.**

The trained arm's 7 qualifying runs already existed when K, the inclusion rule and the
sizing SD were chosen, and they were inspected in the course of choosing them — the SD
oscillates with K (0.153 to 0.034 across K = 100 to 500) and K=400 was picked from that
curve. So:

| element | status |
|---|---|
| the four new control runs | **genuinely prospective** — the rules below were fixed before they produced data |
| the first control run `LR0-20260905T034032Z` | retrospective; it is what motivated the design |
| the 7 trained runs | retrospective; they were used to choose K and to estimate the SD |
| the comparison's overall type-I error | **not controlled at the nominal 0.05** — one arm and the analysis choices were selected with the other arm's data in view |

The correct reading of any result here is: the control arm was collected under fixed
rules, and it is compared against a trained arm that was not. That is weaker than a
pre-registered two-arm trial and must not be reported as one.

Everything below is fixed from this point for the four new control runs; nothing may be
re-chosen after seeing their data.

## Why this document exists

The first analysis of `LR0-20260905T034032Z` chose its comparison prefix (K=400) *after*
looking at the data, and the required sample size depends heavily on that choice. Measured
across the existing 13 `dhv/n=4` runs, the run-to-run SD of Spearman rho does not fall
monotonically with K -- it oscillates:

| K | runs with >=K proposals | SD | runs per arm for delta=0.10 |
|---:|---:|---:|---:|
| 100 | 10 | 0.1527 | 37 |
| 150 | 10 | 0.1068 | 18 |
| 200 | 9 | 0.0715 | 9 |
| 250 | 9 | 0.1090 | 19 |
| 300 | 7 | 0.0909 | 13 |
| 350 | 7 | 0.0422 | 3 |
| 400 | 7 | 0.0552 | 5 |
| 500 | 5 | 0.0341 | 2 |

A post-hoc K can therefore be picked to make the design look as cheap as 2 runs per arm or
as expensive as 37. K=400 sits in a trough. **The estimate that came out of it, "5 per arm,
3 more runs", is not defensible and is withdrawn.**

## Frozen decisions

| item | value | fixed because |
|---|---|---|
| **Cell** | `dhv / n=4` only | the existing control ran there; other cells have no control |
| **Metric** | Spearman rho of `vina` against proposal index | the campaign's existing metric, unchanged |
| **Warm-start** | drop the first 63 proposals | byte-identical prefix in every run, carries no policy signal |
| **Prefix K** | **400** | declared here, before new runs; runs shorter than 400 are excluded |
| **Inclusion** | a run counts iff `--n-samples-per-prompt == 4` read from its own argv, and it produced >= 400 post-warm-start proposals | the argv check already excluded 9 mislabelled runs |
| **Test** | two-sample Welch t on rho, alpha = 0.05 two-sided | |
| **Primary claim tested** | difference in mean rho between frozen-policy and trained arms | |
| **Minimum effect of interest** | **\|delta\| = 0.10** | the trained arm's own mean at K=400 is +0.029, so a difference of 0.10 is more than three times the effect being explained |
| **Equivalence margin, if the result is null** | TOST at \|delta\| < 0.10 | anything narrower is not affordable; see below |
| **Power target** | 0.80 | |

## Sample size, and its honest status

Using SD = 0.0552 (K=400, the 7 existing qualifying runs) gives **5 runs per arm** for
delta = 0.10 at 80% power. The trained arm already has **7 qualifying runs**; the control
arm has **1**. So the shortfall is **4 control runs**, not 3 -- the earlier figure
subtracted the shortfall from the wrong arm.

**That SD is a prior, not a fact about the control arm.** It was estimated from trained
runs, at a K chosen after seeing the data, from 7 samples (so the SD itself carries about
17% relative uncertainty). The analysis will re-estimate SD from the completed arms and
report the achieved resolvable |delta|, whatever it turns out to be. If the re-estimated
SD is materially larger, the correct report is "underpowered at the frozen margin", not a
new sample size chosen to fit.

## Budget

One control run measured **1h50m wall on 3 GPUs = 5.5 GPU-hours**.

| | runs | GPU-hours |
|---|---:|---:|
| already spent (control 1) | 1 | 5.5 |
| **this pre-specification** | **4** | **22** |
| total control arm | 5 | 27.5 |

The trained arm is reused entirely; no new trained runs, and the stopped matrix is not
restarted. Anything beyond these 4 runs is out of budget and must be raised separately.

## What the result can and cannot say

Even fully executed, this design tests one cell of four. It says nothing about the
acquisition objective or about n=2. And a null result licenses only the equivalence claim
at |delta| < 0.10 -- not "no difference".


## Selection effect the inclusion rule creates, recorded 2026-09-05 07:55Z

Applying the frozen rule to the trained arm keeps **7 of 25** `dhv/n=4` run directories.
The 18 exclusions are not evenly spread:

| reason | runs |
|---|---:|
| 0 proposals after warm-start | 8 |
| between 8 and 285 proposals, under K=400 | 5 |
| `argv n_samples == 1` (cannot produce a gradient) | 4 |
| argv unreadable | 1 |

So `K = 400` does two things at once. It matches length, which is what it was chosen for.
It also **filters out most of the early failed and short runs**, leaving the subset that
ran long and ran to completion.

~~The control arm is collected under the same rule, so the comparison is still like-for-like
on this axis.~~ **That does not follow, and it is withdrawn.** Reaching 400 proposals is a
**post-treatment** variable. If the learning rate affects whether a run gets there — through
throughput, through crashes, through anything — then conditioning on `>= 400` selects a
different sub-population in each arm, and applying the same rule to both does not restore
exchangeability. Identical rules do not make selected groups comparable when the selection
criterion is itself downstream of the treatment.

The observed rates are already suggestive and are reported rather than dismissed:

| arm | directories | no gp_history | argv n != 4 | shorter than K | qualifying | reached K |
|---|---:|---:|---:|---:|---:|---:|
| trained, lr=1e-6 | 25 | 0 | 5 | 13 | 7 | **35.0%** |
| control, lr=0 | 2 | 0 | 0 | 0 | 2 | **100%** |

Fisher exact p = 0.156 on 7/20 against 2/2. **That is not evidence of no difference** — with
two control runs the test has almost no power, and the point estimates differ by a factor
of nearly three. Pre-filter lengths: trained `[0,0,0,0,0,0,0,8,20,61,177,253,285,426,493,
501,574,588,619,654]`, control `[523, 842]`.

**Consequence for the main claim, and narrowing the population is not enough.**

~~The claim is narrowed to the population the frozen rule defines, and that makes it valid
for that population.~~ **It does not.** Filtering *each arm* to `>= 400` does not produce
one population observed under two treatments; it produces **two different populations**,
each selected by a criterion that the treatment itself may have influenced. Re-labelling
the target as "runs that reached 400 proposals" does not fix this — "runs that reached 400
proposals **under lr=0**" and "runs that reached 400 proposals **under lr=1e-6**" are not
the same set of runs, and no amount of narrowing makes them one.

So the contrast this design can report is **descriptive**: the difference between two
observed, conditionally-selected groups. It is **not** a treatment effect on a common
population, and it must never be written as one. The wording used throughout is "the
observed difference between the qualifying groups", not "the effect of lr=0".

Recovering a causal contrast would need something this design does not have — an outcome
defined on every run regardless of length (so nothing is conditioned away), or a completion
model that is itself identified. Neither is in scope here.

K is not changed after the fact, and completion rates are reported alongside every number.

This was not written down when the rule was frozen. It is recorded here rather than used
to re-choose K.


## Attempt-level record of the five control runs, 2026-09-05 11:40Z

Every attempt is listed, including the one that failed, because failures are data about the
completion rate and removing them by reason would be selecting on the outcome.

| # | run | proposals | Slurm state | GPU-h | reached K=400 | note |
|---|---|---:|---|---:|---|---|
| 1 | `LR0-20260905T034032Z` | 842 | COMPLETED | 5.50 | yes | |
| 2 | `LR0c2-20260905T072457Z` | 685 | COMPLETED | 5.33 | yes | |
| 3 | `LR0c3-20260905T091137Z` | 670 | COMPLETED | 5.53 | yes | |
| 4 | `LR0c4-20260905T110212Z` | 11 | **FAILED** (exit 1:0, 967 s) | 0.81 | no | `torch.OutOfMemoryError`, tried to allocate 3.70 GiB on a card with a co-tenant |
| 5 | `LR0c5-20260905T111823Z` | running | RUNNING | — | pending | |

**`LR0c4` failed for a reason unrelated to the treatment** — CUDA OOM from contention on a
shared card — and its lr knob had arrived (`--lr 0 --lr-decay-style constant
--weight-decay 0`, and the `[lr-knob]` line is in its log). It is nevertheless **kept in the
denominator** of the completion rate. Dropping it because its cause looks environmental
would be selecting on the outcome, which is the failure this whole section is about. The
cause is recorded so a reader can weigh it; it is not used to adjust the count.

The trained arm's exclusions were never diagnosed to this level — no sacct record survives
for them — so the two arms' completion rates are not decomposed the same way. That is a
further reason the rates are reported as observed quantities rather than treated as an
estimate of anything.


## Final result, 2026-09-05 12:05Z — all five attempts resolved

| # | run | proposals | state | reached K |
|---|---|---:|---|---|
| 1 | `LR0-…034032Z` | 842 | COMPLETED | yes |
| 2 | `LR0c2-…072457Z` | 685 | COMPLETED | yes |
| 3 | `LR0c3-…091137Z` | 670 | COMPLETED | yes |
| 4 | `LR0c4-…110212Z` | 11 | FAILED (CUDA OOM, 16:07) | no |
| 5 | `LR0c5-…111823Z` | 404+ | RUNNING, past K | yes |

Reaching K: **7/20 trained against 4/5 control**, Fisher p = 0.133. Not evidence of equal
completion rates.

**These are three separate verdicts. None implies another.**

**1. Difference — not detected.** Trained +0.0286 (n=7, sd 0.0552) against control +0.0946
(n=4, sd 0.0536); difference −0.0660, 95% CI [−0.1475, +0.0155], Welch t = −1.94, df = 6.5,
**p = 0.0961**. Not significant at the frozen alpha. Note the interval barely contains zero
and all four control runs read positive, so "not detected" is not "absent".

**2. Equivalence — not established, and one-sided only.** TOST at the frozen margin 0.10:
upper p = **0.0011** (the control is not higher than the trained arm by 0.10 or more),
lower p = 0.176 (**cannot** exclude the control being lower by 0.10 or more). One side
holds, the other does not, so equivalence within ±0.10 is not established.

**3. Power — short of the frozen margin.** Pooled SD re-estimates to 0.0544, close to the
0.0552 prior used for sizing. The margin needs 4.6 runs per arm and the smaller arm has 4;
achieved resolvable |delta| is **0.108** against the frozen 0.10. Underpowered, reported as
such, not resized.

**What the numbers are.** A descriptive difference between two conditionally-selected
groups. Not a treatment effect on a common population — the selection criterion is
downstream of the treatment and the arms complete at different observed rates.

**Budget.** 20.1 GPU-hours of the 27.5 authorised across five attempts, with the fifth still running. The 2x2 matrix
remains stopped and no run outside these five was started.