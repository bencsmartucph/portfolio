"""Tests for the built-in resolver ops and check() comparison logic."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

import claimpin
from claimpin.context import Context
from claimpin.resolver import check, resolve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_registry():
    """Reset plugin registries before and after every test."""
    claimpin.clear_registry()
    yield
    claimpin.clear_registry()


@pytest.fixture
def survey_csv(tmp_path):
    """Tiny 4-row CSV for op tests."""
    df = pd.DataFrame({
        "cohort": ["econ", "econ", "polsci", "polsci"],
        "pages": [10.0, 12.0, 8.0, 6.0],
        "score": [3.0, 5.0, None, 7.0],
        "rating": [5.0, 5.0, 3.0, 5.0],  # ceiling = 5.0, two rows hit it
        "item_a": [4.0, 2.0, 3.0, 5.0],
        "item_b": [3.0, 2.0, 4.0, 5.0],
    })
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture
def ctx(tmp_path):
    return Context(tmp_path)


@pytest.fixture
def ctx_with_csv(tmp_path, survey_csv):
    """Context whose project_root contains data.csv."""
    return Context(tmp_path)


@pytest.fixture
def json_file(tmp_path):
    doc = {
        "model": {
            "coef": -0.05,
            "se": 0.01,
            "n": 100,
            "r2": 0.42,
        }
    }
    p = tmp_path / "results.json"
    p.write_text(json.dumps(doc))
    return p


# ---------------------------------------------------------------------------
# check() — comparison logic
# ---------------------------------------------------------------------------

class TestCheck:
    def test_abs_within_tolerance(self):
        claim = {"value": 10.59, "comparison": "abs", "tolerance": 0.005}
        ok, msg = check(claim, 10.592)
        assert ok
        assert "claim=10.59" in msg

    def test_abs_outside_tolerance(self):
        claim = {"value": 10.59, "comparison": "abs", "tolerance": 0.005}
        ok, msg = check(claim, 10.60)
        assert not ok

    def test_tolerance_zero_exact_int(self):
        """tolerance: 0 on integer Ns must not fail on float representation noise."""
        claim = {"value": 240, "comparison": "abs", "tolerance": 0}
        ok, _ = check(claim, 240.0)
        assert ok

    def test_tolerance_zero_epsilon_boundary(self):
        """Even a tiny difference (larger than epsilon) should still fail at tolerance 0."""
        claim = {"value": 240, "comparison": "abs", "tolerance": 0}
        ok, _ = check(claim, 240.001)
        assert not ok

    def test_lt_passes_when_truth_less(self):
        claim = {"value": 0.05, "comparison": "lt"}
        ok, msg = check(claim, 0.03)
        assert ok
        assert "<" in msg

    def test_lt_fails_when_truth_equal(self):
        claim = {"value": 0.05, "comparison": "lt"}
        ok, _ = check(claim, 0.05)
        assert not ok

    def test_lt_fails_when_truth_greater(self):
        claim = {"value": 0.05, "comparison": "lt"}
        ok, _ = check(claim, 0.10)
        assert not ok

    def test_gt_passes_when_truth_greater(self):
        claim = {"value": 0.0, "comparison": "gt"}
        ok, msg = check(claim, 0.5)
        assert ok
        assert ">" in msg

    def test_gt_fails_when_truth_equal(self):
        claim = {"value": 0.0, "comparison": "gt"}
        ok, _ = check(claim, 0.0)
        assert not ok

    def test_gt_fails_when_truth_less(self):
        claim = {"value": 0.5, "comparison": "gt"}
        ok, _ = check(claim, 0.3)
        assert not ok

    def test_default_comparison_is_abs(self):
        """comparison key absent should default to abs."""
        claim = {"value": 5.0}
        ok, _ = check(claim, 5.0)
        assert ok

    def test_message_contains_truth_and_diff(self):
        claim = {"value": 10.0, "comparison": "abs", "tolerance": 0.1}
        _, msg = check(claim, 10.5)
        assert "truth=" in msg
        assert "diff" in msg


# ---------------------------------------------------------------------------
# Built-in resolver ops — JSON
# ---------------------------------------------------------------------------

class TestJsonOps:
    def test_lookup_simple_path(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"a": {"b": 3.14}}))
        ctx = Context(tmp_path)
        binding = {"op": "lookup", "source": "r.json", "path": "a/b"}
        assert resolve(binding, ctx) == pytest.approx(3.14)

    def test_lookup_nested_path(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"x": {"y": {"z": 42}}}))
        ctx = Context(tmp_path)
        binding = {"op": "lookup", "source": "r.json", "path": "x/y/z"}
        assert resolve(binding, ctx) == pytest.approx(42.0)

    def test_z_p_two_sided(self, tmp_path):
        """z = coef/se; p = 2*(1-norm.cdf(|z|)). For z=2 p ≈ 0.0455."""
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"coef": 2.0, "se": 1.0}))
        ctx = Context(tmp_path)
        binding = {"op": "z_p", "source": "r.json",
                   "coef_path": "coef", "se_path": "se"}
        result = resolve(binding, ctx)
        assert result == pytest.approx(0.04550026, abs=1e-5)

    def test_tost_p_equivalence(self, tmp_path):
        """TOST p should be small when |beta| << sesoi."""
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"beta": 0.0, "se": 0.1, "n": 200}))
        ctx = Context(tmp_path)
        binding = {
            "op": "tost_p", "source": "r.json",
            "beta_path": "beta", "se_path": "se", "n_path": "n",
            "sesoi": 0.5,
        }
        result = resolve(binding, ctx)
        # beta=0 with sesoi=0.5 and se=0.1 → t = 5; p should be tiny
        assert result < 0.001


# ---------------------------------------------------------------------------
# Built-in resolver ops — CSV
# ---------------------------------------------------------------------------

class TestCsvOps:
    def test_count_rows(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "count_rows", "source": "data.csv"}
        assert resolve(binding, ctx) == 4.0

    def test_count_where_eq(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "count_where", "source": "data.csv",
                   "where": ["cohort", "==", "econ"]}
        assert resolve(binding, ctx) == 2.0

    def test_count_where_gt(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "count_where", "source": "data.csv",
                   "where": ["pages", ">", 9.0]}
        assert resolve(binding, ctx) == 2.0

    def test_count_where_lt(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "count_where", "source": "data.csv",
                   "where": ["pages", "<", 9.0]}
        assert resolve(binding, ctx) == 2.0

    def test_count_where_lte(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "count_where", "source": "data.csv",
                   "where": ["pages", "<=", 8.0]}
        assert resolve(binding, ctx) == 2.0

    def test_count_where_gte(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "count_where", "source": "data.csv",
                   "where": ["pages", ">=", 10.0]}
        assert resolve(binding, ctx) == 2.0

    def test_col_stat_mean(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "col_stat", "source": "data.csv",
                   "field": "pages", "stat": "mean"}
        assert resolve(binding, ctx) == pytest.approx(9.0)

    def test_col_stat_median(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "col_stat", "source": "data.csv",
                   "field": "pages", "stat": "median"}
        assert resolve(binding, ctx) == pytest.approx(9.0)

    def test_col_stat_with_where(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "col_stat", "source": "data.csv",
                   "field": "pages", "stat": "mean",
                   "where": ["cohort", "==", "econ"]}
        assert resolve(binding, ctx) == pytest.approx(11.0)

    def test_nunique(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "nunique", "source": "data.csv", "col": "cohort"}
        assert resolve(binding, ctx) == 2.0

    def test_notnull_count(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "notnull_count", "source": "data.csv", "col": "score"}
        assert resolve(binding, ctx) == 3.0

    def test_notnull_pct(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "notnull_pct", "source": "data.csv", "col": "score"}
        assert resolve(binding, ctx) == pytest.approx(75.0)

    def test_ceiling_pct(self, tmp_path, survey_csv):
        """Three out of 4 rows have max rating (5.0) → 75%."""
        ctx = Context(tmp_path)
        binding = {"op": "ceiling_pct", "source": "data.csv", "col": "rating"}
        assert resolve(binding, ctx) == pytest.approx(75.0)

    def test_cronbach_alpha(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "cronbach_alpha", "source": "data.csv",
                   "cols": ["item_a", "item_b"]}
        result = resolve(binding, ctx)
        # alpha must be in a plausible range
        assert 0.0 < result < 1.0

    def test_pearson_r(self, tmp_path):
        """Pearson r between two different columns (pages vs item_a)."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        ctx = Context(tmp_path)
        binding = {"op": "pearson_r", "source": "data.csv", "x": "x", "y": "y"}
        result = resolve(binding, ctx)
        assert result == pytest.approx(1.0)

    def test_pearson_p(self, tmp_path):
        """Perfect linear relationship → p should be very small."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0],
                           "y": [2.0, 4.0, 6.0, 8.0, 10.0]})
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        ctx = Context(tmp_path)
        binding = {"op": "pearson_p", "source": "data.csv", "x": "x", "y": "y"}
        result = resolve(binding, ctx)
        assert result < 0.01

    def test_pearson_r2_pct(self, tmp_path):
        """Perfect correlation → R² = 100%."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0],
                           "y": [2.0, 4.0, 6.0, 8.0, 10.0]})
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        ctx = Context(tmp_path)
        binding = {"op": "pearson_r2_pct", "source": "data.csv", "x": "x", "y": "y"}
        result = resolve(binding, ctx)
        assert result == pytest.approx(100.0)

    def test_pearson_coerces_string_label_columns(self, tmp_path):
        """Survey exports ship numeric columns as strings (value labels,
        mixed missing codes); pearson ops coerce and drop the rest."""
        df = pd.DataFrame({
            "x": ["1", "2", "3", "4", "Refusal"],
            "y": ["2", "4", "6", "8", "5"],
        })
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        ctx = Context(tmp_path)
        binding = {"op": "pearson_r", "source": "data.csv", "x": "x", "y": "y"}
        assert resolve(binding, ctx) == pytest.approx(1.0)

    def test_pearson_r_identical_column_rejected(self, tmp_path):
        """x==y is a curation mistake, not a meaningful correlation; the
        resolver rejects it with a readable error (was BUG-001, fixed)."""
        df = pd.DataFrame({"pages": [10.0, 12.0, 8.0, 6.0]})
        p = tmp_path / "data.csv"
        df.to_csv(p, index=False)
        ctx = Context(tmp_path)
        binding = {"op": "pearson_r", "source": "data.csv", "x": "pages", "y": "pages"}
        with pytest.raises(ValueError, match="same column"):
            resolve(binding, ctx)

    def test_lookup_csv_row_filter(self, tmp_path):
        df = pd.DataFrame({"group": ["a", "b"], "val": [1.5, 2.5]})
        p = tmp_path / "t.csv"
        df.to_csv(p, index=False)
        ctx = Context(tmp_path)
        binding = {"op": "lookup", "source": "t.csv",
                   "row": {"group": "b"}, "field": "val"}
        assert resolve(binding, ctx) == pytest.approx(2.5)

    def test_lookup_csv_multiple_matches_raises(self, tmp_path):
        df = pd.DataFrame({"group": ["a", "a"], "val": [1.0, 2.0]})
        p = tmp_path / "t.csv"
        df.to_csv(p, index=False)
        ctx = Context(tmp_path)
        binding = {"op": "lookup", "source": "t.csv",
                   "row": {"group": "a"}, "field": "val"}
        with pytest.raises(ValueError, match="matched 2 rows"):
            resolve(binding, ctx)

    def test_count_where_bad_op_raises(self, tmp_path, survey_csv):
        ctx = Context(tmp_path)
        binding = {"op": "count_where", "source": "data.csv",
                   "where": ["pages", "!=", 10.0]}
        with pytest.raises(ValueError, match="unsupported where op"):
            resolve(binding, ctx)


# ---------------------------------------------------------------------------
# Derived ops
# ---------------------------------------------------------------------------

class TestDerivedOps:
    def test_sum_of_json_lookups(self, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"a": 3.0, "b": 4.0}))
        ctx = Context(tmp_path)
        binding = {
            "op": "sum",
            "terms": [
                {"op": "lookup", "source": "r.json", "path": "a"},
                {"op": "lookup", "source": "r.json", "path": "b"},
            ],
        }
        assert resolve(binding, ctx) == pytest.approx(7.0)

    def test_pct_reduction(self, tmp_path):
        ctx = Context(tmp_path)
        claims_by_id = {
            "claim_a": {"value": 100.0},
            "claim_b": {"value": 70.0},
        }
        binding = {"op": "pct_reduction", "claim_a": "claim_a", "claim_b": "claim_b"}
        result = resolve(binding, ctx, claims_by_id=claims_by_id)
        assert result == pytest.approx(30.0)

    def test_unknown_op_named_before_source_is_touched(self, tmp_path):
        """A typo'd op is reported as such even when the source is also
        missing — the unknown-op check runs before any file access."""
        ctx = Context(tmp_path)
        binding = {"op": "does_not_exist", "source": "nowhere.txt"}
        with pytest.raises(ValueError, match="unknown op 'does_not_exist'"):
            resolve(binding, ctx)

    def test_known_op_wrong_source_family_raises(self, tmp_path):
        (tmp_path / "results.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        ctx = Context(tmp_path)
        binding = {"op": "z_p", "source": "results.csv", "coef_path": "a", "se_path": "b"}
        with pytest.raises(ValueError, match="does not apply to source"):
            resolve(binding, ctx)
