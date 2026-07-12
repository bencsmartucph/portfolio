# Replication code — Dignity Is a Baseline

These are the core analysis scripts behind the seminar paper, published for
audit rather than push-button rerun. The European Social Survey (ESS) and
CWED microdata this pipeline runs on are access-restricted — ESS requires a
free registration and use agreement, CWED requires a data-use request to the
Comparative Welfare Entitlements Dataset team — so the raw data isn't
included here and the scripts won't execute as-is. What's here lets a reader
check the method: how each number in the paper is computed, in what order,
from which inputs.

## What each script does

| Script | Role |
|---|---|
| `run_sorting_mechanism.py` | Builds the master analysis dataset: loads ESS waves 6–9, merges ISCO-08 routine-task-intensity scores, constructs the anti-immigration index and redistribution-support DV, merges CWED welfare-generosity and CPDS macro controls, assigns welfare-regime categories. |
| `final_analysis_pipeline.py` | Historical random-intercepts pipeline. Still canonical for Models 5–6 (redistribution support, radical-right vote) and the country-level welfare-generosity ↔ ALMP-spending comparison. |
| `random_slopes_models.py` | **Canonical source for the paper's headline coefficients.** Fits Models 1–3 with country-wave random slopes on RTI (`1 + task_z | country-wave`), plus BLUPs-based per-country slope extraction and its correlation with CWED generosity. |
| `_diagnose_cwed_correlation.py` | Methodology audit: compares four ways of computing the country-level RTI→attitude slope (bivariate OLS, OLS with controls, BLUPs from the mixed model, country-wave-averaged OLS) to establish why BLUPs is the right estimator for the headline correlation. |
| `reconcile_headline_r_2026-05-29.py` | Locks the headline correlation and its jackknife sensitivity checks to one verbatim computation (copied from `random_slopes_models.py`'s BLUPs block), resolving a rounding discrepancy between the paper text and an earlier notebook pass. |

Order of operations for a full rerun (given the restricted data in place):

```bash
python run_sorting_mechanism.py       # build the master dataset
python final_analysis_pipeline.py     # Models 5-6, ALMP comparison
python random_slopes_models.py        # Models 1-3 (headline), BLUPs correlation
python _diagnose_cwed_correlation.py  # methodology cross-check
python reconcile_headline_r_2026-05-29.py  # locked headline-r reproduction
```

`run_sorting_mechanism.py` and `final_analysis_pipeline.py` read
`RESEARCH_MASTER_ROOT` from the environment (defaulting to the current
directory) rather than a hardcoded path — set it to point at a checkout with
`data/raw/` populated if you have data access.

## Data access

- **ESS** (European Social Survey, rounds 6–9): free registration at
  [europeansocialsurvey.org](https://www.europeansocialsurvey.org/).
- **CWED** (Comparative Welfare Entitlements Dataset): data-use request via
  the CWED project (Lyle Scruggs et al.).
- **CPDS** (Comparative Political Data Set) and the ISCO-08 task-score
  crosswalk are cited in the paper's data appendix; sources there.

No microdata, derived person-level extracts, or `.dta`/`.csv` data files are
included in this folder — code only.

## Key result this pipeline produces

The paper's central empirical finding: welfare *decommodification* (CWED),
not welfare spending, is what explains cross-national variation in how
strongly automation exposure converts into anti-immigration attitudes.

- **Country-level decommodification correlation:** r = −0.855 (N = 15
  countries) — the correlation between CWED generosity and each country's
  BLUP-estimated RTI→anti-immigration slope from the random-slopes model.
- **Individual-level cross-level interaction:** β = −0.059 (Model 3) — the
  RTI × CWED-generosity interaction term: each one-SD increase in welfare
  decommodification reduces the slope of automation exposure on
  anti-immigration attitudes by 0.059 scale points.

Full derivation, controls, and robustness checks are in the paper
([`../Dignity_Is_a_Baseline.docx`](../Dignity_Is_a_Baseline.docx) /
[`../README.md`](../README.md)), sections §V.B–§V.D.
