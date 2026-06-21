"""Quickstart plugin: one custom op and one custom check.

Custom ops cover the paper-specific computations no general tool should try
to guess (here: a group gap defined on a particular analysis sample). Custom
checks cover assertions that are not single-number claims (here: a sign
expectation the theory commits to before seeing the data).
"""
import claimpin


@claimpin.op("cohort_gap")
def cohort_gap(ctx, source, outcome, group_col, group_a, group_b):
    """Mean difference in `outcome` between two groups."""
    df = ctx.load_csv(source)
    a = df[df[group_col] == group_a][outcome].mean()
    b = df[df[group_col] == group_b][outcome].mean()
    return float(a - b)


@claimpin.check("coffee_effect_is_negative")
def coffee_effect_is_negative(ctx):
    """Theory says distance hurts output: the coefficient must be negative."""
    results = ctx.load_json("results/toy_results.json")
    coef = results["model1"]["coef_coffee_distance"]
    assert coef < 0, f"coffee-distance coefficient is non-negative ({coef:.4f}) — sign flip"
