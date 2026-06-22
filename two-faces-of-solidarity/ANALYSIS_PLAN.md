# Analysis plan (frozen) — Two faces of solidarity

> **Status: FROZEN before the confirmatory run.** Written 2026-06-14. This is the pre-commitment
> document: estimand, specification, expected signs, and kill-conditions, fixed before the full-battery
> clustered pipeline runs. **Honesty flag — this is NOT a blind pre-registration.** The hypothesis was
> *exploration-generated*: it came from the 2026-06-12 on-disk probe and a 2026-06-14 coding-direction
> sign-check. What is pre-committed here is the *confirmatory specification* (full controls, country-wave
> clustering, the whole battery) and the falsification conditions, frozen before that specification runs.

## The question

Routine-task exposure (RTI / automation risk) is known to raise anti-immigration attitudes (the parent seminar
paper, RTI×CWED). The paper also reports RTI → **more** redistribution support (β=+0.041, p<0.001) — the
"solidarity pathway." But solidarity has two faces (Cavaillé & Trump 2015; van Oorschot 2006):
- **"How much"** — support for redistribution / generosity. *(the paper measured this)*
- **"For whom"** — who is judged deserving; the conditionality of support. *(unclaimed in the paper)*

**Question.** Do these two faces move the *same* way under RTI, or do they dissociate? Specifically: does
automation exposure raise the *amount* of redistribution demanded while *hardening* the conditions on who
deserves it?

## Data

ESS rounds 6–9 — the merged analysis master file (188,764 obs, 34 countries). The ESS-derived master is restricted and is **not** included in this repo (see the data note in the README).
The welfare-conditionality battery is **wave-8 only** (n≈40k, 23 countries). Redistribution support is
all-wave but is present in wave 8 too, so the **dissociation is tested on the SAME wave-8 respondents**
(joint sample with full controls: n≈32,276).

## Variables — coding pinned to GROUND TRUTH (not to variable names)

The deservingness battery is in **raw ESS coding (1=agree strongly … 5=disagree strongly)**, verified by
the identity check `narrow_deserving == mean(sbstrec, uentrjb)` (max diff 0.000000) and by anchoring each
item against `redist_support` (high=pro-redistribution; `redist_support × lrscale = −0.166`). **Orientation
is decided by the anchors, never by the name** — `narrow_deserving` is *mis*-named (high = INCLUSIVE).

| Variable | Item / construction | Raw direction | Oriented so HIGHER = MORE INCLUSIVE |
|---|---|---|---|
| `task_z` | RTI = std(`task`), 3-digit ISCO-08 | higher = more routine | — (predictor) |
| `redist_support` | `gincdif` reverse-coded, 1–5 | higher = more pro-redistribution | "how much" outcome (already oriented) |
| `uentrjb` | "Most unemployed don't really try" | 1=agree(restrictive)…5=disagree | use raw (higher=inclusive) |
| `sbstrec` | "Benefits strain economy" | 1=agree(restrictive)…5=disagree | use raw (higher=inclusive) |
| `sbbsntx` | "Benefits cost business too much" | 1=agree(restrictive)…5=disagree | use raw (higher=inclusive) |
| `sbprvpv` | "Benefits prevent poverty" (PRO-worded) | 1=agree(inclusive)…5=disagree | **REVERSE** (6−x) so higher=inclusive |
| `narrow_deserving` | `mean(sbstrec, uentrjb)`, raw | higher=inclusive | use raw |

**Index to build + validate:** `deserving_inclusive` = mean of oriented {uentrjb, sbstrec, sbbsntx,
reverse(sbprvpv)}, higher = more inclusive/less-conditional. Report Cronbach's α; report each item
separately too. Primary "for whom" measures: the conditionality/reciprocity items `uentrjb` and
`narrow_deserving` (cleanest deservingness signal); `sbstrec`/`sbbsntx`/`sbprvpv` (welfare-consequence
items) are corroborating, reported separately.

**Controls:** age, age², female, education (`college` and/or `eisced`), household income (`hinctnta`),
urban. **Clustering:** country-wave (in wave 8, country = country-wave → 23 clusters; use cluster-robust
SE **and** wild-cluster bootstrap, per the few-clusters rule).

## The coding bug this corrects (verified, first-class)

The "higher = more restrictive" label for `narrow_deserving` is **backwards** (it is higher = more inclusive),
and the error already propagated:
- the construction script — its comment reads "higher = more restrictive".
- the codebook — same mislabel.
- the parent paper's analysis report — Model 7's `RTI→deservingness = −0.030` is flagged
  "**(unexpected direction)**" and `deservingness→anti-immig = −0.453` "**(unexpected direction — reflects the
  scaling of the deservingness measure)**." Both dissolve under the corrected coding: −0.030 = RTI → MORE
  restrictive (expected backlash); −0.453 = inclusive-deservingness → less anti-immig (expected). The mediation
  **RTI → more restrictive deservingness → more anti-immigration** is clean and currently mis-read as a puzzle.
- NOT in the signed manuscript (the seminar paper does not report deservingness) — no signed work is wrong.

This correction is documented here with the wrong-vs-corrected reading. The fix to the parent
pipeline is applied separately; this repo only records the correction.

## The sbprvpv coherence check (built-in internal-validity test)

`sbprvpv` ("benefits prevent poverty") is the one pro-welfare-*worded* item — reverse polarity to the others.
If "higher RTI → more restrictive" is a real uniform signal, the four negatively-worded items
(`uentrjb`, `sbstrec`, `sbbsntx`, raw `narrow_deserving`) must take **negative** task_z coefficients while
`sbprvpv` **flips positive**. The probe already shows exactly this (uentrjb −0.092, sbstrec −0.022, sbbsntx
−0.042, **sbprvpv +0.024**). This sign-flip is the discriminator: it flipping = uniform backlash holds; it NOT
flipping = the dimensions genuinely dissociate (also shippable, a different claim). Report it explicitly.

## Specifications (frozen)

0. **S0 — Verify the bug to ground truth.** Confirm the three mislabel sites and reproduce the parent analysis report's
   Model 7 reading of −0.030/−0.453 as "(unexpected direction)". Document the correction.
1. **S1 — Reproduce ground truth (verification gate).** Re-estimate model7 step1
   (`narrow_deserving ~ task_z + controls`, wave-8, cntry_wave RE) and confirm it matches
   the parent paper's committed ground-truth result (coef ≈ **−0.0297**, p≈1.3e-07). Re-estimate RTI→`redist_support` with the paper's
   controls and confirm ≈ **+0.041**. If these don't reproduce, the pipeline is wrong — stop and fix.
2. **S2 — Primary dissociation (the headline).** On the wave-8 joint sample, estimate within-country-wave
   (FE, cluster-robust + wild bootstrap):
   - `redist_support ~ task_z + controls`  → expect **β > 0**
   - `deserving_inclusive ~ task_z + controls`  → expect **β < 0**
   - Dissociation confirmed iff both hold (opposite signs, both distinguishable from 0).
3. **S3 — Battery decomposition.** Item-by-item RTI→{each oriented item}; report the sign pattern and which
   items drive it.
4. **S4 — Robustness.** (a) OLS+FE vs mixed model; (b) education/income in vs out (does the deservingness
   arm survive the key confounders?); (c) ordinal (ologit) for the single-item outcomes to address scale
   compression; (d) standardized effect sizes so the two arms are comparable despite different scales.
5. **S5 — Is it one latent dimension?** Factor / correlation structure of redist_support vs the
   deservingness items — do they load separately (a real two-dimensional structure) or is the
   "dissociation" noise on one latent factor?

## Expected signs (frozen, exploration-generated)

`RTI → redist_support`: **positive**. `RTI → deserving_inclusive`: **negative**. Dissociation: both, in the
same wave-8 respondents, under full controls + country-wave clustering.

## What would kill this (first-class deliverable — `WHAT_WOULD_KILL_THIS.md`)

1. **Confounding.** If `RTI→deserving_inclusive` loses significance or flips when education+income enter →
   the deservingness arm is a class/income artifact, not an RTI effect. **Shippable null.**
2. **Measurement artifact.** `redist_support` is single-item, ceiling-bound (31% at max, per the paper);
   the deservingness items are better dispersed. A sign *difference* could partly reflect scale compression.
   Must be addressed (ordinal models, standardized effects) — if the dissociation only exists on the raw
   linear scale and vanishes under ordinal/standardized treatment, it is not safe.
3. **One-dimension collapse.** If redist_support and the deservingness items load on a single factor, the
   "two faces" framing is wrong.
4. **Reverse causality / occupational selection.** RTI is not randomly assigned; cross-sectional design
   cannot establish direction. Stated, not solved.
5. **Construct validity.** `sbstrec`/`sbbsntx` measure perceived *consequences* of welfare more than
   *deservingness* per se; the cleanest "for whom" signal is `uentrjb`/`narrow_deserving`. If the
   dissociation rests only on the consequence items and not the conditionality items, the deservingness
   framing weakens.

## Estimator note (don't inherit the scouts' "house spec")

The shipped M2–M7 are `mixedlm` random-intercept models, NOT OLS+FE. For the wave-8 deservingness models,
country-wave collapses to country → **23 clusters**, so: **OLS + country fixed effects + cluster-robust SE,
plus wild-cluster bootstrap** (Rademacher/Webb) for the few-clusters regime — the standard few-clusters toolkit
(Cameron, Gelbach & Miller 2008). Report both the analytic-cluster and wild-bootstrap p. For the ground-truth reproduction (S1), match the
shipped estimator (`mixedlm`) to hit −0.0297 exactly.

## Deliverables

1. `code/pipeline.py` — reproduces S0–S5; writes `results/numbers.csv` (every cited number).
2. `code/sur_dissociation.py` — the stacked/SUR joint dissociation test.
3. `results/numbers.csv`, `results/sur_numbers.csv` — the canonical numbers behind every claim.
4. `WHAT_WOULD_KILL_THIS.md` — the falsification audit, answered against the actual results.
5. `README.md` — what this is, the data note, and the honest finding (the finding writeup is Ben's to author).

## Honesty rails

- Every number in the writeup regenerates into `results/numbers.csv` from `code/` and matches. A claim
  that can't reproduce is a failed run.
- The finding writeup is Ben's to author; this plan and the code are the reproducible substrate behind it.
- A null (any of the kill-conditions firing) is a SHIPPABLE SUCCESS — the honest re-analysis is the point.
- The coding-direction correction (anchor on `redist_support`, not the variable name) is the methodological
  contribution and must be shown explicitly.
