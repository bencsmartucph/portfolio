"""Tests for schema.py: load_claims_file and validate."""
from __future__ import annotations

import textwrap

import pytest
import yaml

from claimpin.schema import ClaimsError, load_claims_file, validate


def write_yaml(tmp_path, content: str):
    p = tmp_path / "claims.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


class TestValidate:
    def test_valid_bound_claim(self):
        claims = [{"id": "c1", "value": 10, "binds_to": {"op": "count_rows", "source": "f.csv"}}]
        validate(claims)  # should not raise

    def test_valid_review_claim(self):
        claims = [{"id": "c1", "value": 10, "review": True}]
        validate(claims)  # no binds_to needed when review: true

    def test_duplicate_id_raises(self):
        claims = [
            {"id": "c1", "value": 10, "binds_to": {"op": "count_rows", "source": "f.csv"}},
            {"id": "c1", "value": 20, "binds_to": {"op": "count_rows", "source": "f.csv"}},
        ]
        with pytest.raises(ClaimsError, match="duplicate id"):
            validate(claims)

    def test_missing_id_raises(self):
        claims = [{"value": 10, "binds_to": {"op": "count_rows", "source": "f.csv"}}]
        with pytest.raises(ClaimsError, match="missing `id`"):
            validate(claims)

    def test_missing_value_raises(self):
        claims = [{"id": "c1", "binds_to": {"op": "count_rows", "source": "f.csv"}}]
        with pytest.raises(ClaimsError, match="missing `value`"):
            validate(claims)

    def test_missing_binds_to_without_review_raises(self):
        claims = [{"id": "c1", "value": 10}]
        with pytest.raises(ClaimsError, match="no `binds_to`"):
            validate(claims)

    def test_bad_comparison_raises(self):
        claims = [{"id": "c1", "value": 10, "comparison": "ne",
                   "binds_to": {"op": "count_rows", "source": "f.csv"}}]
        with pytest.raises(ClaimsError, match="comparison.*not in"):
            validate(claims)

    def test_binds_to_not_mapping_raises(self):
        claims = [{"id": "c1", "value": 10, "binds_to": "count_rows"}]
        with pytest.raises(ClaimsError, match="`binds_to` must be a mapping"):
            validate(claims)

    def test_claims_not_list_raises(self):
        with pytest.raises(ClaimsError, match="must be a list"):
            validate({"not": "a list"})

    def test_non_dict_claim_raises(self):
        with pytest.raises(ClaimsError, match="is not a mapping"):
            validate(["not_a_dict"])

    def test_valid_comparisons_all_accepted(self):
        for comp in ("abs", "lt", "gt"):
            claims = [{"id": "c1", "value": 10, "comparison": comp,
                       "binds_to": {"op": "count_rows", "source": "f.csv"}}]
            validate(claims)  # no exception


class TestLoadClaimsFile:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ClaimsError, match="not found"):
            load_claims_file(tmp_path / "nonexistent.yaml")

    def test_no_claims_key_raises(self, tmp_path):
        p = tmp_path / "claims.yaml"
        p.write_text("meta:\n  note: nothing\n", encoding="utf-8")
        with pytest.raises(ClaimsError, match="top-level `claims` list"):
            load_claims_file(p)

    def test_valid_file_loads(self, tmp_path):
        p = write_yaml(tmp_path, """
            meta:
              note: test
            claims:
              - id: c1
                value: 10
                review: true
        """)
        doc = load_claims_file(p)
        assert len(doc["claims"]) == 1
        assert doc["claims"][0]["id"] == "c1"

    def test_meta_defaulted_when_absent(self, tmp_path):
        p = write_yaml(tmp_path, """
            claims:
              - id: c1
                value: 5
                review: true
        """)
        doc = load_claims_file(p)
        assert doc["meta"] == {}
