# WHAT_WOULD_KILL_THIS.md — the falsification audit, answered against results

> The five kill-conditions from the frozen plan (`ANALYSIS_PLAN.md` §"What would
> kill this"), each answered against the confirmatory run. Numbers regenerate
> into `results/numbers.csv`. Headline: **two of the kill-conditions fired.** The
> deservingness arm is not robust to education controls and is not distinguishable
> from zero by wild bootstrap under full controls. This is a SHIPPABLE NULL on the
> dissociation, paired with a CLEAN coherence result. Report it as such.

## Verdict table

| # | Kill-condition | Fired? | Evidence |
|---|---|---|---|
| 1 | Confounding (education/income) | **YES** | deserving arm β goes −0.044 (p_webb .002) → **−0.012 (p_webb .169)** when `college` enters |
| 2 | Measurement artifact / scale compression | NO (survives) | ordinal + standardized estimates preserve the sign pattern |
| 3 | One-dimension collapse | NO (two factors) | 2 eigenvalues > 1; redist & deservingness load separately; but battery α=0.46 |
| 4 | Reverse causality / occupational selection | UNRESOLVED (by design) | cross-sectional; stated, not solved |
| 5 | Construct validity (consequence vs conditionality items) | PARTIAL | the cleanest item (`uentrjb`) is the strongest; consequence items weaker |

---

## 1. Confounding — **FIRED (the decisive one)**

The deservingness arm is largely an **education artifact**, not a clean RTI effect.
`deserving_inclusive ~ task_z` with country FE:

- no controls: β = −0.0453 (p_webb = 0.0011, n = 38,861)
- + demographics (age, age², female): β = −0.0439 (p_webb = 0.0019, n = 38,750)
- **+ college: β = −0.0124 (p_webb = 0.1685, n = 38,624)** ← collapses
- + income (full controls): β = −0.0178 (p_webb = 0.0630, n = 32,611)

Adding `college` to the demographics frame (same ~38.6k rows, apples-to-apples)
absorbs roughly 70% of the deservingness effect and pushes it well below
conventional significance. The redistribution arm, by contrast, is stable
and significant throughout (β = +0.084 → +0.049 across the same ladder, every
p_webb ≤ 0.0007). **This is the cleanest reading of the run:** RTI robustly raises
redistribution demand; its effect on inclusive deservingness is mostly a class
gradient that routine-task exposure proxies. The asymmetry the paper found on the
*moderation* side has a parallel here on the *level* side — but the deservingness
half does not stand on its own once class is controlled.

## 2. Measurement artifact / scale compression — does NOT fire

`redist_support` is single-item and ceiling-bound; the concern is that a sign
*difference* between arms is an artifact of different scale shapes. Addressed two ways:

- **Ordinal (ologit + country dummies):** redist +0.089 (p 7e-14); uentrjb −0.070
  (p 2e-9); sbstrec −0.023 (p .050); sbbsntx −0.033 (p .005); sbprvpv +0.011 (p .34).
  Same sign pattern as the linear models — the dissociation in *signs* is not a
  linear-scale artifact.
- **Standardized (outcome z-scored):** redist +0.049 SD vs deserving_inclusive
  −0.027 SD per 1 SD RTI. The redist arm is ~1.8× the deservingness arm in SD units;
  the gap is real but modest.

The *sign pattern* survives. What does not survive is the deservingness arm's
*significance* under full controls (kill-condition 1), which is the binding issue.

## 3. One-dimension collapse — does NOT fire (but the battery is weak)

PCA on the 5 oriented items: eigenvalues [1.71, 1.03, 0.98, 0.76, 0.53] → **two
factors with eigenvalue > 1** (Kaiser). `redist_support` loads weakly on PC1
(0.255) while the three negatively-worded conditionality/consequence items load
strongly (0.61–0.80); `sbprvpv_rev` is essentially its own factor (PC2 = 0.948).
Mean r(redist, deservingness items) = 0.061 vs within-deservingness r = 0.172.

So the data are **not** one latent dimension — the "two faces" framing is not
refuted by the factor structure. BUT: Cronbach α for `deserving_inclusive` = 0.46,
which is poor. The four items do not form a tight scale (sbprvpv barely coheres
with the rest even after reversal). The "for whom" construct is measured noisily,
which itself limits power on the deservingness arm.

## 4. Reverse causality / occupational selection — UNRESOLVED by design

RTI is not randomly assigned. People may sort into routine jobs for reasons
correlated with both redistribution demand and deservingness views (the standard
ESS occupational-selection critique). Cross-sectional ESS cannot establish
direction. Stated, not solved — same limitation the parent paper carries.

## 5. Construct validity — PARTIAL

The cleanest "for whom" / reciprocity item is `uentrjb` ("most unemployed people do
not really try to find a job"), and it is the **strongest and only individually
wild-bootstrap-significant** battery item: β = −0.040, p_webb = 0.012. `narrow_deserving` (the
mean of sbstrec+uentrjb) is −0.026 (p_webb = 0.055). The pure welfare-*consequence*
items are weaker: sbstrec −0.012 (p_webb .35), sbbsntx −0.018 (p_webb .076). So the
deservingness signal is carried by the conditionality item, not the consequence
items — which is the *right* direction for the "for whom" interpretation, but means
the index as a whole is diluted by the noisier consequence items.

---

## Bottom line

The headline dissociation (RTI raises redistribution demand **and** hardens
deservingness, jointly, under full controls + clustering) **does not survive** at
p < .05 on the deservingness arm by wild bootstrap, and the deservingness arm is
substantially an education artifact (kill-condition 1). What **does** survive and
is clean:

1. **The coding correction** (S0): `narrow_deserving` higher = inclusive; Model 7's
   two "(unexpected direction)" flags dissolve into an ordinary chauvinism pathway.
2. **The sign pattern + coherence check** (S3): all four negatively-worded items
   take negative RTI coefficients and the pro-worded `sbprvpv` flips positive — a
   coherent "RTI → more restrictive" direction, even if only `uentrjb` is
   individually significant.
3. **The redistribution arm** (S2): RTI robustly raises redistribution support,
   wave-8 within-country, every spec.

The honest claim is narrower than the frozen hypothesis: **RTI reliably raises the
*amount* of redistribution demanded; it pushes deservingness views in the
restrictive direction in sign but that push is weak and largely explained by
education.** A directional dissociation in signs, not a clean dual-significant one.
