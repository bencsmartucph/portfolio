"""Binding resolver: turns a claim's `binds_to` dict into a ground-truth number.

Built-in op families:

  derived (no source)      sum, pct_reduction
  JSON sources (*.json)    lookup, z_p, tost_p
  tabular sources (*.csv)  lookup, count_rows, count_where, col_stat, nunique,
                           notnull_count, notnull_pct, ceiling_pct,
                           cronbach_alpha, pearson_r, pearson_p, pearson_r2_pct

Anything else dispatches to the project's custom-op registry (see ops.py
plugins, registered with @claimpin.op). Custom ops are consulted first, so a
project can override a built-in when its semantics genuinely differ — document
it in the plugin when you do.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import claimpin

from .context import Context


def _json_path(doc: dict, path: str):
    node = doc
    for key in path.split("/"):
        node = node[key]
    return node


def _csv_rows(df: pd.DataFrame, row_filter: dict | None) -> pd.DataFrame:
    if not row_filter:
        return df
    mask = pd.Series(True, index=df.index)
    for col, val in row_filter.items():
        mask &= df[col].astype(str) == str(val)
    return df[mask]


def _where_mask(df: pd.DataFrame, where: list) -> pd.Series:
    col, op, val = where
    series = df[col]
    ops = {
        "<": series.lt, "<=": series.le,
        ">": series.gt, ">=": series.ge,
        "==": series.eq,
    }
    if op not in ops:
        raise ValueError(f"unsupported where op: {op}")
    return ops[op](val)


def cronbach_alpha(items: pd.DataFrame) -> float:
    items = items.dropna()
    k = items.shape[1]
    item_var = items.var(ddof=1).sum()
    total_var = items.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var / total_var))


DERIVED_OPS = {"sum", "pct_reduction"}
JSON_OPS = {"lookup", "z_p", "tost_p"}
CSV_OPS = {"lookup", "count_rows", "count_where", "col_stat", "nunique", "notnull_count",
           "notnull_pct", "ceiling_pct", "cronbach_alpha", "pearson_r", "pearson_p",
           "pearson_r2_pct"}


def resolve(binds_to: dict, ctx: Context, claims_by_id: dict | None = None) -> float:
    """Return the ground-truth value for a claim binding."""
    op = binds_to.get("op", "lookup")
    source = binds_to.get("source")

    # Unknown op named BEFORE any source is touched, so a typo'd op is never
    # masked by a missing-file error.
    known = DERIVED_OPS | JSON_OPS | CSV_OPS | set(claimpin.OPS)
    if op not in known:
        custom = f"; custom: {sorted(claimpin.OPS)}" if claimpin.OPS else ""
        raise ValueError(f"unknown op {op!r} (built-ins: {sorted(DERIVED_OPS | JSON_OPS | CSV_OPS)}{custom})")

    # ── custom ops (project plugin) ───────────────────────────────────────
    if op in claimpin.OPS:
        params = {k: v for k, v in binds_to.items() if k != "op"}
        return float(claimpin.OPS[op](ctx, **params))

    # ── derived ops (no source) ───────────────────────────────────────────
    if op == "sum":
        return sum(resolve(term, ctx, claims_by_id) for term in binds_to["terms"])
    if op == "pct_reduction":
        # Percentage reduction between two CLAIMED values: an internal-
        # consistency check on the prose, deliberately not a recomputation.
        a = claims_by_id[binds_to["claim_a"]]["value"]
        b = claims_by_id[binds_to["claim_b"]]["value"]
        return (1.0 - b / a) * 100.0

    # ── JSON sources ──────────────────────────────────────────────────────
    if source and source.endswith(".json"):
        doc = ctx.load_json(source)
        if op == "lookup":
            return float(_json_path(doc, binds_to["path"]))
        if op == "z_p":
            coef = float(_json_path(doc, binds_to["coef_path"]))
            se = float(_json_path(doc, binds_to["se_path"]))
            z = coef / se
            return float(2 * (1 - stats.norm.cdf(abs(z))))
        if op == "tost_p":
            # Two one-sided t-tests against |beta| = sesoi, df = n - 1.
            beta = float(_json_path(doc, binds_to["beta_path"]))
            se = float(_json_path(doc, binds_to["se_path"]))
            n = int(_json_path(doc, binds_to["n_path"]))
            sesoi = float(binds_to["sesoi"])
            df_t = n - 1
            t_lower = (beta + sesoi) / se
            t_upper = (beta - sesoi) / se
            p_lower = float(1 - stats.t.cdf(t_lower, df_t))
            p_upper = float(stats.t.cdf(t_upper, df_t))
            return max(p_lower, p_upper)

    # ── tabular sources ───────────────────────────────────────────────────
    if source and source.endswith(".csv"):
        df = ctx.load_csv(source)
        if op == "lookup":
            rows = _csv_rows(df, binds_to.get("row"))
            if len(rows) != 1:
                raise ValueError(
                    f"row filter {binds_to.get('row')} matched {len(rows)} rows in {source}"
                )
            return float(rows.iloc[0][binds_to["field"]])
        if op == "count_rows":
            return float(len(df))
        if op == "count_where":
            return float(_where_mask(df, binds_to["where"]).sum())
        if op == "col_stat":
            sub = df
            if "where" in binds_to:
                sub = df[_where_mask(df, binds_to["where"])]
            series = sub[binds_to["field"]].dropna()
            return float(getattr(series, binds_to["stat"])())
        if op == "nunique":
            return float(df[binds_to["col"]].nunique())
        if op == "notnull_count":
            return float(df[binds_to["col"]].notnull().sum())
        if op == "notnull_pct":
            return float(df[binds_to["col"]].notnull().mean() * 100.0)
        if op == "ceiling_pct":
            col = df[binds_to["col"]].dropna()
            return float((col == col.max()).mean() * 100.0)
        if op == "cronbach_alpha":
            return cronbach_alpha(df[binds_to["cols"]])
        if op in ("pearson_r", "pearson_p", "pearson_r2_pct"):
            if binds_to["x"] == binds_to["y"]:
                raise ValueError(f"{op}: x and y are the same column ({binds_to['x']!r})")
            # Survey exports often ship numeric columns as strings (value
            # labels, mixed missing codes); coerce so non-numeric entries
            # become NaN and fall out with the existing dropna.
            sub = df[[binds_to["x"], binds_to["y"]]].apply(pd.to_numeric, errors="coerce").dropna()
            r, p = stats.pearsonr(sub[binds_to["x"]], sub[binds_to["y"]])
            if op == "pearson_r":
                return float(r)
            if op == "pearson_p":
                return float(p)
            return float(r * r * 100.0)

    raise ValueError(
        f"unresolvable binding: op {op!r} does not apply to source {source!r} "
        f"(expected a .json/.csv source matching the op family, or a custom op)"
    )


def check(claim: dict, truth: float) -> tuple[bool, str]:
    """Compare a claim against ground truth. Returns (ok, message)."""
    value = claim["value"]
    comparison = claim.get("comparison", "abs")
    if comparison == "lt":
        ok = truth < value
        return ok, f"truth={truth:.6g} {'<' if ok else '>='} bound={value}"
    if comparison == "gt":
        ok = truth > value
        return ok, f"truth={truth:.6g} {'>' if ok else '<='} bound={value}"
    # The 1e-12 epsilon keeps tolerance: 0 claims (exact integer Ns) from
    # failing on float representation noise.
    tolerance = float(claim.get("tolerance", 0.0)) + 1e-12
    diff = abs(float(value) - float(truth))
    ok = diff <= tolerance
    return ok, f"claim={value} truth={truth:.6g} |diff|={diff:.2e} tol={claim.get('tolerance', 0)}"
