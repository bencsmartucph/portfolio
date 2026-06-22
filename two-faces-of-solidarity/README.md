# Two faces of solidarity

A small confirmatory study, run as a methods-and-integrity exercise: does routine-task exposure
(RTI) move the *two faces* of redistributive preference — "how much" (support for redistribution)
and "for whom" (the conditionality of who is judged deserving) — in the same direction, or do they
dissociate? It extends the parent seminar paper *Dignity Is a Baseline*.

What this folder is built to show is not a headline result. It is a way of working:

- **A frozen pre-commitment.** [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) fixes the estimand,
  specification, expected signs, and kill-conditions *before* the confirmatory run — and is honest
  that the hypothesis was exploration-generated, not blindly pre-registered.
- **A falsification audit, answered against the data.** [`WHAT_WOULD_KILL_THIS.md`](WHAT_WOULD_KILL_THIS.md)
  takes the five pre-committed kill-conditions and reports which ones *fired* — including a
  conceded null on the deservingness arm under education controls.
- **Reproducible code.** [`code/pipeline.py`](code/pipeline.py) implements OLS + country fixed
  effects with cluster-robust SEs and a from-scratch wild-cluster bootstrap (Webb 6-point and
  Rademacher) for the 23-cluster regime; [`code/sur_dissociation.py`](code/sur_dissociation.py)
  reframes the dissociation as a cross-equation restriction via a stacked/SUR test. Every cited
  number regenerates into [`results/numbers.csv`](results/numbers.csv).

## Data note

The analysis runs on a dataset derived from the European Social Survey (rounds 6–9), merged with
welfare-generosity and occupational task-intensity measures. **That dataset is restricted and is
not included here** — ESS microdata redistribution is governed by use agreements, and merging does
not confer redistribution rights. The code is provided to show the method; it is not runnable
as-is. Only derived, aggregate results (coefficients, standard errors, p-values) are reported.

## The finding

Routine-task exposure moves the two faces of redistributive preference — but not together. The *how much* face responds cleanly: greater RTI reliably raises support for redistribution, and the effect is stable across the full control ladder (β from +0.084 with no controls down to +0.049 under demographics, education, and income — every wild-bootstrap *p* ≤ 0.0007). The *for whom* face is where the story narrows. In sign, more exposure does track more restrictive deservingness views — but the effect is weak and does not stand on its own. A single education control absorbs roughly 70 percent of it and pushes it below conventional significance (β −0.044, *p* = .002 → −0.012, *p* = .169); under full controls it sits at −0.018 (*p* = .063). What looked like a hardening of who counts as deserving is largely a class gradient that routine-task exposure proxies.

So the honest result is a *directional* dissociation, not the cleaner dual-significant one the frozen hypothesis hoped for: the amount of redistribution demanded rises with exposure; the conditionality of deservingness moves the same restrictive way in sign but is too weak — and too entangled with education — to claim on its own. The sign pattern is at least coherent (all four negatively-worded items take negative coefficients; the pro-worded item flips positive), and the two faces are empirically distinct rather than one latent dimension (two factors with eigenvalue > 1). But the deservingness battery is noisy (α = 0.46), and only the sharpest conditionality item — "most unemployed people don't really try to find a job" — is individually significant (β −0.040, *p* = .012).

Two of the five pre-committed kill-conditions fired, and this folder reports them as such. That is the exercise: the plan named in advance what would sink the claim, the audit ran those tests against the data, and two of them landed. The conceded null is the finding.
