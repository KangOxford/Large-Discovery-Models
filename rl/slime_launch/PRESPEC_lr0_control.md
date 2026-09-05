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

**Consequence for the main claim.** Any result from this comparison is about **runs that
reached 400 proposals**, not about runs of this configuration in general. K is not changed
after the fact; the claim is narrowed to the population the frozen rule actually defines,
and the completion rates are reported alongside it every time.

This was not written down when the rule was frozen. It is recorded here rather than used
to re-choose K.
