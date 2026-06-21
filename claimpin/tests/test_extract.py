"""Tests for extract.py: parse_number, scan, extract, audit."""
from __future__ import annotations

import pytest

from claimpin.extract import (
    KEYWORD_RE,
    NUMBER_RE,
    audit,
    extract,
    parse_number,
    scan,
)


# ---------------------------------------------------------------------------
# parse_number — edge cases
# ---------------------------------------------------------------------------

class TestParseNumber:
    def test_plain_integer(self):
        assert parse_number("240") == 240.0

    def test_plain_float(self):
        assert parse_number("10.59") == pytest.approx(10.59)

    def test_thousands_separator(self):
        """'188,764' → 188764 (comma as group separator)."""
        assert parse_number("188,764") == pytest.approx(188764.0)

    def test_decimal_comma(self):
        """'0,86' → 0.86 (European decimal comma)."""
        assert parse_number("0,86") == pytest.approx(0.86)

    def test_unicode_minus(self):
        """'−0.059' (unicode minus U+2212) → -0.059."""
        result = parse_number("−0.059")
        assert result == pytest.approx(-0.059)

    def test_negative_integer(self):
        assert parse_number("-42") == -42.0

    def test_both_comma_and_dot(self):
        """'1,234.5' → 1234.5 (comma stripped as group separator)."""
        result = parse_number("1,234.5")
        assert result == pytest.approx(1234.5)

    def test_ambiguous_multi_comma_returns_none(self):
        """'1,23,4' — two commas, second part not length 3 → None."""
        result = parse_number("1,23,4")
        assert result is None

    def test_scientific_notation(self):
        result = parse_number("3e5")
        assert result == pytest.approx(300000.0)

    def test_plain_year_passes_through(self):
        """parse_number itself does NOT exclude years; that is scan's job."""
        assert parse_number("2026") == 2026.0

    def test_single_digit(self):
        assert parse_number("0") == 0.0


# ---------------------------------------------------------------------------
# scan — year exclusion and keyword filtering
# ---------------------------------------------------------------------------

class TestScan:
    def test_year_excluded(self):
        lines = ["Data collected in 2026, N=45 observations"]
        hits = scan(lines)
        raws = [h["raw"] for h in hits]
        assert "2026" not in raws
        assert "45" in raws

    def test_no_keyword_line_yields_no_hits(self):
        lines = ["This line has numbers 3 and 7 but no keywords."]
        hits = scan(lines)
        assert hits == []

    def test_keyword_line_yields_hits(self):
        lines = ["The mean is 10.5 and the std is 2.3"]
        hits = scan(lines)
        assert len(hits) == 2

    def test_multiple_numbers_on_one_line(self):
        lines = ["N=100, mean=5.5, SD=1.2"]
        hits = scan(lines)
        raws = [h["raw"] for h in hits]
        assert "100" in raws
        assert "5.5" in raws
        assert "1.2" in raws

    def test_line_numbers_are_1_indexed(self):
        lines = ["no keywords here", "N=50 students", "N=60 more"]
        hits = scan(lines)
        assert all(h["line"] >= 1 for h in hits)
        line_nums = {h["line"] for h in hits}
        assert 2 in line_nums
        assert 3 in line_nums

    def test_unicode_minus_number_captured(self):
        lines = ["The coefficient β=−0.059 is negative"]
        hits = scan(lines)
        raws = [h["raw"] for h in hits]
        assert "−0.059" in raws


# ---------------------------------------------------------------------------
# extract — skeleton generation
# ---------------------------------------------------------------------------

class TestExtract:
    def test_extract_creates_file(self, tmp_path):
        ms = tmp_path / "paper.md"
        ms.write_text("N=50 students with mean=3.5\n", encoding="utf-8")
        out = tmp_path / "claims.yaml"
        doc = extract(ms, out)
        assert out.exists()
        assert doc["meta"]["n_claims"] > 0

    def test_all_stubs_are_review_true(self, tmp_path):
        ms = tmp_path / "paper.md"
        ms.write_text("N=100 observations, mean=4.2\n", encoding="utf-8")
        out = tmp_path / "claims.yaml"
        doc = extract(ms, out)
        assert all(c["review"] is True for c in doc["claims"])

    def test_refuses_to_overwrite_existing_claims(self, tmp_path):
        """A curated claims file must never be destroyed by a re-run of
        extract; force=True is the explicit opt-in."""
        ms = tmp_path / "paper.md"
        ms.write_text("N=50 students with mean=3.5\n", encoding="utf-8")
        out = tmp_path / "claims.yaml"
        out.write_text("meta: {}\nclaims: []\n", encoding="utf-8")
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            extract(ms, out)
        assert out.read_text(encoding="utf-8") == "meta: {}\nclaims: []\n"
        doc = extract(ms, out, force=True)
        assert doc["meta"]["n_claims"] > 0

    def test_year_not_extracted(self, tmp_path):
        ms = tmp_path / "paper.md"
        ms.write_text("Collected in 2026, N=75 respondents\n", encoding="utf-8")
        out = tmp_path / "claims.yaml"
        doc = extract(ms, out)
        values = [c["value"] for c in doc["claims"]]
        assert 2026.0 not in values
        assert 75.0 in values

    def test_meta_contains_sha256(self, tmp_path):
        ms = tmp_path / "paper.md"
        ms.write_text("N=10 participants\n", encoding="utf-8")
        out = tmp_path / "claims.yaml"
        doc = extract(ms, out)
        assert "manuscript_sha256" in doc["meta"]
        assert len(doc["meta"]["manuscript_sha256"]) == 64

    def test_unique_ids_generated(self, tmp_path):
        ms = tmp_path / "paper.md"
        # Two numbers on the same line → different ids
        ms.write_text("mean=5.5 and SD=1.2 and N=30\n", encoding="utf-8")
        out = tmp_path / "claims.yaml"
        doc = extract(ms, out)
        ids = [c["id"] for c in doc["claims"]]
        assert len(ids) == len(set(ids))

    def test_custom_keyword_regex(self, tmp_path):
        ms = tmp_path / "paper.md"
        ms.write_text("CUSTOM 42 apples\nnormal line 99\n", encoding="utf-8")
        out = tmp_path / "claims.yaml"
        doc = extract(ms, out, keyword_regex="CUSTOM")
        values = [c["value"] for c in doc["claims"]]
        assert 42.0 in values
        assert 99.0 not in values


# ---------------------------------------------------------------------------
# audit — drift and coverage
# ---------------------------------------------------------------------------

class TestAudit:
    def _make_claims_doc(self, manuscript_path, sha256, claims):
        return {
            "meta": {
                "manuscript": str(manuscript_path),
                "manuscript_sha256": sha256,
            },
            "claims": claims,
        }

    def test_no_drift_clean_state(self, tmp_path):
        ms = tmp_path / "paper.md"
        ms.write_text("The mean is 5.5 observations\n", encoding="utf-8")
        import hashlib
        sha = hashlib.sha256(ms.read_bytes()).hexdigest()
        doc = self._make_claims_doc(ms, sha, [
            {
                "id": "c1",
                "text_snippet": "The mean is 5.5 observations",
                "manuscript_lines": [1],
                "value": 5.5,
                "review": True,
            }
        ])
        result = audit(doc, ms)
        assert result["hash_drift"] is None
        assert result["drifted_claims"] == []
        assert result["ok"] is True

    def test_hash_drift_detected(self, tmp_path):
        ms = tmp_path / "paper.md"
        ms.write_text("N=50 respondents\n", encoding="utf-8")
        doc = self._make_claims_doc(ms, "deadbeef" * 8, [])
        result = audit(doc, ms)
        assert result["hash_drift"] is not None
        assert result["hash_drift"]["recorded"] == "deadbeef" * 8

    def test_prose_drift_detected(self, tmp_path):
        """Claim says '5.5' is on line 1, but line 1 now says '6.0'."""
        ms = tmp_path / "paper.md"
        ms.write_text("The mean is 6.0 observations\n", encoding="utf-8")
        import hashlib
        sha = hashlib.sha256(ms.read_bytes()).hexdigest()
        doc = self._make_claims_doc(ms, sha, [
            {
                "id": "c1",
                "text_snippet": "The mean is 5.5 observations",
                "manuscript_lines": [1],
                "value": 5.5,
                "review": True,
            }
        ])
        result = audit(doc, ms)
        assert len(result["drifted_claims"]) == 1
        assert result["drifted_claims"][0]["id"] == "c1"
        assert result["ok"] is False

    def test_unbound_lines_reported(self, tmp_path):
        """A keyword line with a number but no claim covering it → unbound."""
        ms = tmp_path / "paper.md"
        ms.write_text("N=75 respondents participated\n", encoding="utf-8")
        import hashlib
        sha = hashlib.sha256(ms.read_bytes()).hexdigest()
        doc = self._make_claims_doc(ms, sha, [])
        result = audit(doc, ms)
        assert any(u["line"] == 1 for u in result["unbound_lines"])

    def test_covered_line_not_in_unbound(self, tmp_path):
        ms = tmp_path / "paper.md"
        ms.write_text("N=75 respondents participated\n", encoding="utf-8")
        import hashlib
        sha = hashlib.sha256(ms.read_bytes()).hexdigest()
        doc = self._make_claims_doc(ms, sha, [
            {
                "id": "c1",
                "text_snippet": "N=75 respondents",
                "manuscript_lines": [1],
                "value": 75,
                "review": True,
            }
        ])
        result = audit(doc, ms)
        assert result["unbound_lines"] == []
