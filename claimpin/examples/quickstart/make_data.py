"""Regenerate the quickstart's synthetic data and results (deterministic).

The committed survey.csv / toy_results.json were produced by this script;
rerun it only if you mean to rebuild the example from scratch.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(42)

n = 240
coffee_distance = rng.uniform(5, 300, n)  # metres to nearest coffee machine
focus = rng.normal(5, 1.5, n)
pages_per_week = np.clip(8 - 0.012 * coffee_distance + 0.9 * focus + rng.normal(0, 1.2, n), 0, None)
supervisor_meetings = rng.poisson(2, n)
cohort = rng.choice(["econ", "polsci"], n)

# three Likert items forming a "thesis dread" scale
latent = rng.normal(0, 1, n)
items = {f"dread_{i}": np.clip(np.round(3 + latent + rng.normal(0, 0.6, n)), 1, 5) for i in (1, 2, 3)}

df = pd.DataFrame({
    "coffee_distance": coffee_distance.round(1),
    "focus": focus.round(2),
    "pages_per_week": pages_per_week.round(2),
    "supervisor_meetings": supervisor_meetings,
    "cohort": cohort,
    **items,
})
df.to_csv(HERE / "results" / "survey.csv", index=False, encoding="utf-8")

# "the analysis": one OLS fit, saved like a pipeline would save it
import statsmodels.formula.api as smf

fit = smf.ols("pages_per_week ~ coffee_distance + focus", data=df).fit()
results = {
    "model1": {
        "coef_coffee_distance": float(fit.params["coffee_distance"]),
        "se_coffee_distance": float(fit.bse["coffee_distance"]),
        "coef_focus": float(fit.params["focus"]),
        "n": int(fit.nobs),
        "r2": float(fit.rsquared),
    }
}
with open(HERE / "results" / "toy_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1)

print(json.dumps(results, indent=1))
print("mean pages:", df["pages_per_week"].mean())
print("r(coffee, pages):", df["coffee_distance"].corr(df["pages_per_week"]))
print("econ rows:", (df["cohort"] == "econ").sum())
print("gap econ-polsci:", df[df.cohort == "econ"].pages_per_week.mean() - df[df.cohort == "polsci"].pages_per_week.mean())
