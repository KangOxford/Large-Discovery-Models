# Pre-specification: additional lr=0 control runs

Frozen 2026-09-05 06:00Z, **before** any additional control run is launched. Everything
below is fixed from this point; nothing here may be re-chosen after seeing new data.

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
