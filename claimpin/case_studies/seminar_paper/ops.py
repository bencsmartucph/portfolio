"""Per-project plugin for the seminar-paper case study.

Ports, nearly verbatim, the paper-specific parts of
Research_Master/verification/ (the golden-parity oracle):

  custom ops    the master_* recomputations from tests/resolver.py — they
                encode the pipeline's analysis-sample definition, which no
                general tool should guess
  custom checks the hand-written tests from test_tier_a_recompute.py (Model 1
                refit) and test_tier_a_sanity.py (theory-aware direction,
                plausibility, and degeneracy assertions)

Check names mirror the oracle's test function names (minus the `test_`
prefix) so the parity script can map them one-to-one.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import claimpin

MASTER_CSV = "analysis/sorting_mechanism_master_v2.csv"

# Pipeline conventions (final_analysis_pipeline.py §4): complete cases on the
# base model variables, regime "Other" excluded.
PIPELINE_CONTROLS = ["agea", "age_sq", "female", "college", "hinctnta", "urban"]
PIPELINE_BASE_VARS = ["anti_immig_index", "task_z", "welfare_regime", "cntry_wave"] + PIPELINE_CONTROLS
REGIME_ORDER = ["Nordic", "Continental", "Liberal", "Southern", "Eastern"]

MODEL1_FORMULA = (
    "anti_immig_index ~ task_z + agea + age_sq + female + college "
    "+ hinctnta + urban + C(cntry_wave)"
)

RS_MODELS = ["M1_baseline_rs", "M2_regime_rs", "M3_cwed_rs", "M4_education_rs",
             "M5_redistribution_rs"]


def pipeline_analysis_sample(master_df: pd.DataFrame) -> pd.DataFrame:
    sample = master_df.dropna(subset=PIPELINE_BASE_VARS)
    return sample[sample["welfare_regime"].isin(REGIME_ORDER)]


def _bivar_slope(data: pd.DataFrame) -> float:
    sub = data[["anti_immig_index", "task_z"]].dropna()
    slope, _ = np.polyfit(sub["task_z"].values, sub["anti_immig_index"].values, 1)
    return float(slope)


# ── custom binding ops (master-CSV recomputations) ───────────────────────────

@claimpin.op("master_count_rows")
def master_count_rows(ctx, source):
    return float(len(ctx.load_csv(source)))


@claimpin.op("master_nunique")
def master_nunique(ctx, source, col):
    return float(ctx.load_csv(source)[col].nunique())


@claimpin.op("master_notnull_count")
def master_notnull_count(ctx, source, col):
    return float(ctx.load_csv(source)[col].notnull().sum())


@claimpin.op("master_notnull_pct")
def master_notnull_pct(ctx, source, col):
    return float(ctx.load_csv(source)[col].notnull().mean() * 100.0)


@claimpin.op("master_ceiling_pct")
def master_ceiling_pct(ctx, source, col):
    series = ctx.load_csv(source)[col].dropna()
    return float((series == series.max()).mean() * 100.0)


@claimpin.op("master_cronbach")
def master_cronbach(ctx, source, cols):
    from claimpin.resolver import cronbach_alpha
    return cronbach_alpha(ctx.load_csv(source)[cols])


@claimpin.op("master_regime_bivar_slope")
def master_regime_bivar_slope(ctx, source, regime):
    sample = pipeline_analysis_sample(ctx.load_csv(source))
    return _bivar_slope(sample[sample["welfare_regime"] == regime])


@claimpin.op("master_liberal_edu_slope")
def master_liberal_edu_slope(ctx, source, college):
    sample = pipeline_analysis_sample(ctx.load_csv(source))
    lib = sample[sample["welfare_regime"] == "Liberal"]
    return _bivar_slope(lib[lib["college"] == college])


# ── shared fixtures for the checks ───────────────────────────────────────────

_MODEL1 = None


def _model1_refit(ctx):
    """Re-fit Model 1 exactly as final_analysis_pipeline.py §4 does (~15s, cached)."""
    global _MODEL1
    if _MODEL1 is None:
        sample = pipeline_analysis_sample(ctx.load_csv(MASTER_CSV))
        _MODEL1 = smf.ols(MODEL1_FORMULA, data=sample).fit(
            cov_type="cluster", cov_kwds={"groups": sample["cntry_wave"]}
        )
    return _MODEL1


def _final_results(ctx):
    return ctx.load_json("analysis/final_results.json")


def _rs_results(ctx):
    return ctx.load_csv("outputs/tables/rs_results.csv")


def _walk_numbers(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_numbers(v, f"{path}/{k}")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, float(node)


# ── Model 1 refit vs committed results (stale-artifact catch) ────────────────

@claimpin.check("model1_coef_recomputes")
def model1_coef_recomputes(ctx):
    got = float(_model1_refit(ctx).params["task_z"])
    want = _final_results(ctx)["model1"]["coef_rti"]
    assert abs(got - want) < 1e-6, (
        f"Model 1 RTI coefficient: refit {got:.8f} vs final_results.json {want:.8f} "
        f"— final_results.json is stale relative to the master CSV"
    )


@claimpin.check("model1_se_recomputes")
def model1_se_recomputes(ctx):
    got = float(_model1_refit(ctx).bse["task_z"])
    want = _final_results(ctx)["model1"]["se_rti"]
    assert abs(got - want) < 1e-6, (
        f"Model 1 cluster-robust SE (cntry_wave): refit {got:.8f} vs "
        f"final_results.json {want:.8f}"
    )


@claimpin.check("model1_n_recomputes")
def model1_n_recomputes(ctx):
    refit = _model1_refit(ctx)
    assert int(refit.nobs) == _final_results(ctx)["model1"]["n"]


@claimpin.check("model1_r2_recomputes")
def model1_r2_recomputes(ctx):
    got = float(_model1_refit(ctx).rsquared)
    want = _final_results(ctx)["model1"]["r2"]
    assert abs(got - want) < 1e-6


# ── outcome coding direction (the single most common deadline error) ─────────

@claimpin.check("redist_support_is_reverse_coded_gincdif")
def redist_support_is_reverse_coded_gincdif(ctx):
    sub = ctx.load_csv(MASTER_CSV)[["gincdif", "redist_support"]].dropna()
    r = sub["gincdif"].corr(sub["redist_support"])
    assert r < 0, (
        f"redist_support correlates POSITIVELY with raw gincdif (r={r:.3f}) — "
        f"outcome coding direction is wrong; recheck before touching any theory"
    )
    assert r < -0.99, f"expected a deterministic reverse-coding, got r={r:.3f}"


@claimpin.check("anti_immig_items_aligned")
def anti_immig_items_aligned(ctx):
    items = ctx.load_csv(MASTER_CSV)[["imwbcnt", "imueclt", "imbgeco"]].dropna()
    corr = items.corr()
    for a in corr.columns:
        for b in corr.columns:
            assert corr.loc[a, b] > 0, f"{a} vs {b} correlate negatively: index misaligned"


# ── expected signs (the paper's headline claims) ─────────────────────────────

@claimpin.check("rti_main_effect_positive")
def rti_main_effect_positive(ctx):
    final_results, rs_results = _final_results(ctx), _rs_results(ctx)
    assert final_results["model1"]["coef_rti"] > 0
    assert final_results["model3"]["coef_rti"] > 0
    for model in ["M1_baseline_rs", "M2_regime_rs", "M3_cwed_rs"]:
        coef = float(rs_results.set_index("model").loc[model, "rti_coef"])
        assert coef > 0, f"{model} RTI main effect is non-positive ({coef:.4f})"


@claimpin.check("interaction_negative")
def interaction_negative(ctx):
    final_results, rs_results = _final_results(ctx), _rs_results(ctx)
    ri = final_results["model3"]["coef_interaction"]
    rs = float(rs_results.set_index("model").loc["M3_cwed_rs", "int_task_z:cwed_generosity_z"])
    assert ri < 0, f"RI model3 interaction is non-negative ({ri:.4f}) — HEADLINE SIGN FLIP"
    assert rs < 0, f"RS M3 interaction is non-negative ({rs:.4f}) — HEADLINE SIGN FLIP"


# ── magnitude plausibility ───────────────────────────────────────────────────

@claimpin.check("interaction_smaller_than_main_effect")
def interaction_smaller_than_main_effect(ctx):
    final_results, rs_results = _final_results(ctx), _rs_results(ctx)
    assert abs(final_results["model3"]["coef_interaction"]) < abs(final_results["model3"]["coef_rti"])
    m3 = rs_results.set_index("model").loc["M3_cwed_rs"]
    assert abs(float(m3["int_task_z:cwed_generosity_z"])) < abs(float(m3["rti_coef"]))


@claimpin.check("coefficients_in_plausible_band")
def coefficients_in_plausible_band(ctx):
    final_results, rs_results = _final_results(ctx), _rs_results(ctx)
    for path, val in _walk_numbers(final_results):
        leaf = path.rsplit("/", 1)[-1]
        if leaf.startswith("coef") or leaf == "r":
            assert abs(val) < 5, f"{path} = {val} outside plausible band"
    coef_cols = ["rti_coef"] + [c for c in rs_results.columns
                                if c.startswith("int_task_z:") and not c.endswith(("_se", "_p"))]
    for col in coef_cols:
        for val in rs_results[col].dropna():
            assert abs(float(val)) < 5, f"rs_results {col} = {val} outside plausible band"


@claimpin.check("sample_sizes_in_plausible_band")
def sample_sizes_in_plausible_band(ctx):
    final_results, rs_results = _final_results(ctx), _rs_results(ctx)
    assert 50_000 <= final_results["model3"]["n"] <= 120_000
    rs = rs_results.set_index("model")
    assert 50_000 <= int(rs.loc["M3_cwed_rs", "n_obs"]) <= 120_000
    for model in ["M1_baseline_rs", "M2_regime_rs", "M4_education_rs", "M5_redistribution_rs"]:
        assert 100_000 <= int(rs.loc[model, "n_obs"]) <= 200_000, f"{model} n out of band"
    for path, val in _walk_numbers(final_results):
        if path.rsplit("/", 1)[-1] == "n":
            assert val > 0, f"{path} non-positive n"


@claimpin.check("pvalues_are_probabilities")
def pvalues_are_probabilities(ctx):
    final_results, rs_results = _final_results(ctx), _rs_results(ctx)
    for path, val in _walk_numbers(final_results):
        leaf = path.rsplit("/", 1)[-1]
        if leaf == "p" or leaf.startswith("p_"):
            assert 0 <= val <= 1, f"{path} = {val} is not a probability"
    p_cols = [c for c in rs_results.columns if c.endswith(("_p", "pval"))]
    for col in p_cols:
        for val in rs_results[col].dropna():
            assert 0 <= float(val) <= 1, f"rs_results {col} = {val} is not a probability"


@claimpin.check("all_models_converged")
def all_models_converged(ctx):
    for model, node in _final_results(ctx).items():
        if isinstance(node, dict) and "converged" in node:
            assert node["converged"] is True, f"{model} did not converge"


# ── scale reliability ────────────────────────────────────────────────────────

@claimpin.check("cronbach_alpha_matches_documented")
def cronbach_alpha_matches_documented(ctx):
    from claimpin.resolver import cronbach_alpha
    alpha = cronbach_alpha(ctx.load_csv(MASTER_CSV)[["imwbcnt", "imueclt", "imbgeco"]])
    assert abs(alpha - 0.864) < 0.001, f"Cronbach alpha recomputes to {alpha:.4f}, documented 0.864"


# ── multilevel degeneracy guard (econometrics rules §2) ──────────────────────

def _register_rs_variance_checks():
    for model in RS_MODELS:
        def check_fn(ctx, model=model):
            var = _rs_results(ctx).set_index("model").loc[model, "int_task_z Var"]
            var = float(var)
            assert not math.isnan(var), f"{model}: slope variance component missing"
            assert var > 1e-6, f"{model}: slope variance {var} is degenerate (collapsed to zero)"
        check_fn.__doc__ = f"Random-slope model {model} keeps a non-degenerate slope-variance component."
        claimpin.check(f"random_slope_variance_nonzero[{model}]")(check_fn)


_register_rs_variance_checks()
