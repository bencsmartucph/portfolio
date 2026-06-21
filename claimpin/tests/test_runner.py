"""Tests for runner.py: verify orchestration, statuses, plugin registry."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd
import pytest
import yaml

import claimpin
from claimpin.runner import verify


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clean_registry():
    claimpin.clear_registry()
    yield
    claimpin.clear_registry()


def make_project(tmp_path: Path, csv_rows=None, json_doc=None):
    """Set up a minimal project directory for runner tests."""
    if csv_rows is not None:
        df = pd.DataFrame(csv_rows)
        df.to_csv(tmp_path / "data.csv", index=False)
    if json_doc is not None:
        (tmp_path / "results.json").write_text(json.dumps(json_doc), encoding="utf-8")


def write_claims(tmp_path: Path, claims_list: list, meta: dict | None = None) -> Path:
    doc = {"meta": meta or {}, "claims": claims_list}
    p = tmp_path / "claims.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Status: pass
# ---------------------------------------------------------------------------

class TestPass:
    def test_simple_pass(self, tmp_path):
        make_project(tmp_path, csv_rows={"x": [1, 2, 3]})
        claims_path = write_claims(tmp_path, [
            {"id": "c1", "value": 3, "comparison": "abs", "tolerance": 0,
             "binds_to": {"op": "count_rows", "source": "data.csv"}}
        ])
        result = verify(claims_path)
        assert result["ok"] is True
        assert result["summary"]["pass"] == 1
        assert result["summary"]["fail"] == 0

    def test_pass_with_tolerance(self, tmp_path):
        make_project(tmp_path, csv_rows={"pages": [10.5, 10.7]})
        claims_path = write_claims(tmp_path, [
            {"id": "mean_pages", "value": 10.6, "comparison": "abs", "tolerance": 0.05,
             "binds_to": {"op": "col_stat", "source": "data.csv",
                          "field": "pages", "stat": "mean"}}
        ])
        result = verify(claims_path)
        assert result["ok"] is True


# ---------------------------------------------------------------------------
# Status: fail
# ---------------------------------------------------------------------------

class TestFail:
    def test_simple_fail(self, tmp_path):
        make_project(tmp_path, csv_rows={"x": [1, 2, 3]})
        claims_path = write_claims(tmp_path, [
            {"id": "c1", "value": 99, "comparison": "abs", "tolerance": 0,
             "binds_to": {"op": "count_rows", "source": "data.csv"}}
        ])
        result = verify(claims_path)
        assert result["ok"] is False
        assert result["summary"]["fail"] == 1
        item = next(it for it in result["items"] if it["id"] == "c1")
        assert item["status"] == "fail"

    def test_lt_fail(self, tmp_path):
        make_project(tmp_path, csv_rows={"x": [1, 2, 3]})
        claims_path = write_claims(tmp_path, [
            # truth = 3.0, bound = 2 → truth is NOT < 2 → fail
            {"id": "c1", "value": 2, "comparison": "lt",
             "binds_to": {"op": "count_rows", "source": "data.csv"}}
        ])
        result = verify(claims_path)
        assert result["ok"] is False

    def test_gt_fail(self, tmp_path):
        make_project(tmp_path, csv_rows={"x": [1, 2, 3]})
        claims_path = write_claims(tmp_path, [
            # truth = 3.0, bound = 5 → truth is NOT > 5 → fail
            {"id": "c1", "value": 5, "comparison": "gt",
             "binds_to": {"op": "count_rows", "source": "data.csv"}}
        ])
        result = verify(claims_path)
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# Status: skip
# ---------------------------------------------------------------------------

class TestSkip:
    def test_review_claim_is_skipped(self, tmp_path):
        claims_path = write_claims(tmp_path, [
            {"id": "c1", "value": 30, "review": True,
             "notes": "no artifact yet"}
        ])
        result = verify(claims_path)
        assert result["ok"] is True  # skip does not make run fail
        assert result["summary"]["skip"] == 1
        item = next(it for it in result["items"] if it["id"] == "c1")
        assert item["status"] == "skip"
        assert "REVIEW" in item["message"]

    def test_skip_message_contains_notes(self, tmp_path):
        claims_path = write_claims(tmp_path, [
            {"id": "c1", "value": 30, "review": True,
             "notes": "awaiting unpublished replication"}
        ])
        result = verify(claims_path)
        item = next(it for it in result["items"] if it["id"] == "c1")
        assert "awaiting unpublished replication" in item["message"]


# ---------------------------------------------------------------------------
# Status: error
# ---------------------------------------------------------------------------

class TestError:
    def test_missing_source_file_gives_error(self, tmp_path):
        claims_path = write_claims(tmp_path, [
            {"id": "c1", "value": 3, "comparison": "abs", "tolerance": 0,
             "binds_to": {"op": "count_rows", "source": "nonexistent.csv"}}
        ])
        result = verify(claims_path)
        assert result["ok"] is False
        assert result["summary"]["error"] == 1
        item = next(it for it in result["items"] if it["id"] == "c1")
        assert item["status"] == "error"

    def test_error_does_not_abort_run(self, tmp_path):
        """One error claim + one passing claim: the passing one still shows pass."""
        make_project(tmp_path, csv_rows={"x": [1]})
        claims_path = write_claims(tmp_path, [
            {"id": "bad", "value": 0, "comparison": "abs", "tolerance": 0,
             "binds_to": {"op": "count_rows", "source": "missing.csv"}},
            {"id": "good", "value": 1, "comparison": "abs", "tolerance": 0,
             "binds_to": {"op": "count_rows", "source": "data.csv"}},
        ])
        result = verify(claims_path)
        statuses = {it["id"]: it["status"] for it in result["items"]}
        assert statuses["bad"] == "error"
        assert statuses["good"] == "pass"

    def test_error_item_has_trace(self, tmp_path):
        claims_path = write_claims(tmp_path, [
            {"id": "c1", "value": 3, "comparison": "abs", "tolerance": 0,
             "binds_to": {"op": "count_rows", "source": "missing.csv"}}
        ])
        result = verify(claims_path)
        item = result["items"][0]
        assert "trace" in item


# ---------------------------------------------------------------------------
# Plugin registry — custom @claimpin.op and @claimpin.check
# ---------------------------------------------------------------------------

class TestPlugin:
    def _write_ops(self, tmp_path, content: str) -> Path:
        p = tmp_path / "ops.py"
        p.write_text(textwrap.dedent(content), encoding="utf-8")
        return p

    def test_custom_op_runs(self, tmp_path):
        make_project(tmp_path, csv_rows={"val": [10, 20, 30]})
        ops_path = self._write_ops(tmp_path, """
            import claimpin

            @claimpin.op("triple_count")
            def triple_count(ctx, source):
                df = ctx.load_csv(source)
                return float(len(df) * 3)
        """)
        claims_path = write_claims(tmp_path, [
            {"id": "c1", "value": 9, "comparison": "abs", "tolerance": 0,
             "binds_to": {"op": "triple_count", "source": "data.csv"}}
        ], meta={"ops_module": "ops.py"})
        result = verify(claims_path)
        assert result["ok"] is True
        assert result["items"][0]["status"] == "pass"

    def test_custom_check_pass(self, tmp_path):
        make_project(tmp_path, json_doc={"coef": -0.05})
        ops_path = self._write_ops(tmp_path, """
            import claimpin

            @claimpin.check("sign_check")
            def sign_check(ctx):
                \"\"\"Coefficient must be negative.\"\"\"
                doc = ctx.load_json("results.json")
                assert doc["coef"] < 0, "sign flip"
        """)
        claims_path = write_claims(tmp_path, [], meta={"ops_module": "ops.py"})
        result = verify(claims_path)
        items = {it["id"]: it for it in result["items"]}
        assert "sign_check" in items
        assert items["sign_check"]["status"] == "pass"

    def test_custom_check_fail_on_assertion(self, tmp_path):
        make_project(tmp_path, json_doc={"coef": 0.05})  # positive — should fail
        ops_path = self._write_ops(tmp_path, """
            import claimpin

            @claimpin.check("sign_check")
            def sign_check(ctx):
                \"\"\"Coefficient must be negative.\"\"\"
                doc = ctx.load_json("results.json")
                assert doc["coef"] < 0, f"sign flip: {doc['coef']}"
        """)
        claims_path = write_claims(tmp_path, [], meta={"ops_module": "ops.py"})
        result = verify(claims_path)
        items = {it["id"]: it for it in result["items"]}
        assert items["sign_check"]["status"] == "fail"
        assert result["ok"] is False

    def test_custom_check_error_on_exception(self, tmp_path):
        """Non-AssertionError in a check → status error, not crash."""
        ops_path = self._write_ops(tmp_path, """
            import claimpin

            @claimpin.check("broken_check")
            def broken_check(ctx):
                \"\"\"This check raises unexpectedly.\"\"\"
                raise RuntimeError("unexpected failure")
        """)
        claims_path = write_claims(tmp_path, [], meta={"ops_module": "ops.py"})
        result = verify(claims_path)
        items = {it["id"]: it for it in result["items"]}
        assert items["broken_check"]["status"] == "error"

    def test_clear_registry_between_tests(self, tmp_path):
        """Registry should be empty after clear_registry — no stale ops."""
        assert "my_op" not in claimpin.OPS
        claimpin.OPS["my_op"] = lambda ctx: 1.0
        claimpin.clear_registry()
        assert "my_op" not in claimpin.OPS

    def test_duplicate_op_registration_raises(self):
        claimpin.OPS["dup_op"] = lambda ctx: 1.0
        with pytest.raises(ValueError, match="duplicate custom op"):
            @claimpin.op("dup_op")
            def dup_op(ctx):
                return 1.0

    def test_duplicate_check_registration_raises(self):
        claimpin.CHECKS["dup_check"] = lambda ctx: None
        with pytest.raises(ValueError, match="duplicate custom check"):
            @claimpin.check("dup_check")
            def dup_check(ctx):
                pass

    def test_verify_result_structure(self, tmp_path):
        """verify() result has the expected top-level keys."""
        claims_path = write_claims(tmp_path, [
            {"id": "c1", "value": 5, "review": True}
        ])
        result = verify(claims_path)
        for key in ("claims_file", "project_root", "meta", "items", "summary", "ok"):
            assert key in result

    def test_summary_counts_all_statuses(self, tmp_path):
        make_project(tmp_path, csv_rows={"x": [1]})
        claims_path = write_claims(tmp_path, [
            {"id": "p", "value": 1, "comparison": "abs", "tolerance": 0,
             "binds_to": {"op": "count_rows", "source": "data.csv"}},
            {"id": "f", "value": 99, "comparison": "abs", "tolerance": 0,
             "binds_to": {"op": "count_rows", "source": "data.csv"}},
            {"id": "s", "value": 5, "review": True},
            {"id": "e", "value": 3, "comparison": "abs", "tolerance": 0,
             "binds_to": {"op": "count_rows", "source": "missing.csv"}},
        ])
        result = verify(claims_path)
        s = result["summary"]
        assert s["pass"] == 1
        assert s["fail"] == 1
        assert s["skip"] == 1
        assert s["error"] == 1
