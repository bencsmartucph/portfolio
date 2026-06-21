"""End-to-end CLI tests: subprocess and cli.main(argv) against quickstart."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

import claimpin
from claimpin.cli import main


QUICKSTART = Path(__file__).parent.parent / "examples" / "quickstart"


@pytest.fixture(autouse=True)
def clean_registry():
    claimpin.clear_registry()
    yield
    claimpin.clear_registry()


# ---------------------------------------------------------------------------
# Quickstart end-to-end: expect exit 0
# ---------------------------------------------------------------------------

class TestQuickstartCLI:
    def test_verify_quickstart_exit_0(self):
        """claimpin verify on the real quickstart should exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "claimpin.cli", "verify",
             str(QUICKSTART / "claims.yaml")],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"Expected exit 0 but got {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_verify_quickstart_summary_counts(self):
        """13 passed, 0 failed, 0 errored, 1 skipped."""
        result = subprocess.run(
            [sys.executable, "-m", "claimpin.cli", "verify",
             str(QUICKSTART / "claims.yaml"), "--quiet"],
            capture_output=True, text=True,
        )
        assert "13 passed" in result.stdout
        assert "0 failed" in result.stdout
        assert "1 skipped" in result.stdout

    def test_verify_quickstart_via_main(self):
        """Run via cli.main(argv) — same expectation, no subprocess."""
        rc = main([
            "verify",
            str(QUICKSTART / "claims.yaml"),
        ])
        assert rc == 0

    def test_verify_quickstart_json_output(self, tmp_path):
        out = tmp_path / "report.json"
        rc = main([
            "verify",
            str(QUICKSTART / "claims.yaml"),
            "--json", str(out),
        ])
        assert rc == 0
        assert out.exists()
        doc = json.loads(out.read_text())
        assert doc["ok"] is True
        assert doc["summary"]["pass"] == 13


# ---------------------------------------------------------------------------
# Mutation: copy quickstart to tmp_path, mutate, expect exit 1
# ---------------------------------------------------------------------------

class TestMutationCLI:
    @pytest.fixture
    def mutated_quickstart(self, tmp_path):
        """Copy quickstart to tmp_path, mutate mean_pages claim to wrong value."""
        shutil.copytree(QUICKSTART, tmp_path / "qs")
        qs = tmp_path / "qs"

        # Mutate claims.yaml: change mean_pages value from 10.59 to 10.95
        claims_file = qs / "claims.yaml"
        with open(claims_file, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        for c in doc["claims"]:
            if c["id"] == "mean_pages":
                c["value"] = 10.95
        with open(claims_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)

        # Mutate paper.md: change "averages 10.59 pages" to "averages 10.95 pages"
        paper = qs / "paper.md"
        text = paper.read_text(encoding="utf-8")
        paper.write_text(text.replace("averages 10.59 pages", "averages 10.95 pages"),
                         encoding="utf-8")
        return qs

    def test_mutated_exits_1(self, mutated_quickstart):
        rc = main([
            "verify",
            str(mutated_quickstart / "claims.yaml"),
        ])
        assert rc == 1

    def test_mutated_shows_exactly_one_failure(self, mutated_quickstart, capsys):
        main([
            "verify",
            str(mutated_quickstart / "claims.yaml"),
        ])
        out = capsys.readouterr().out
        assert "mean_pages" in out
        # Other claims still pass — only one FAIL line
        fail_lines = [ln for ln in out.splitlines() if "FAIL" in ln]
        assert len(fail_lines) == 1

    def test_mutated_json_report_identifies_failure(self, mutated_quickstart, tmp_path):
        out = tmp_path / "report.json"
        main([
            "verify",
            str(mutated_quickstart / "claims.yaml"),
            "--json", str(out),
        ])
        doc = json.loads(out.read_text())
        assert doc["ok"] is False
        failed = [it for it in doc["items"] if it["status"] == "fail"]
        assert len(failed) == 1
        assert failed[0]["id"] == "mean_pages"


# ---------------------------------------------------------------------------
# CLI: extract and audit subcommands
# ---------------------------------------------------------------------------

class TestExtractCLI:
    def test_extract_creates_yaml(self, tmp_path):
        ms = tmp_path / "paper.md"
        ms.write_text("N=50 respondents with mean=3.5\n", encoding="utf-8")
        out = tmp_path / "claims.yaml"
        rc = main(["extract", str(ms), "-o", str(out)])
        assert rc == 0
        assert out.exists()

    def test_extract_exit_0(self, tmp_path):
        ms = tmp_path / "paper.md"
        ms.write_text("N=10 observations\n", encoding="utf-8")
        rc = main(["extract", str(ms), "-o", str(tmp_path / "c.yaml")])
        assert rc == 0


class TestAuditCLI:
    def test_audit_clean_state_exit_0(self, tmp_path):
        """Extract then immediately audit → should be clean (no hash drift)."""
        ms = tmp_path / "paper.md"
        ms.write_text("N=50 respondents\n", encoding="utf-8")
        claims_out = tmp_path / "claims.yaml"
        main(["extract", str(ms), "-o", str(claims_out)])
        rc = main([
            "audit", str(claims_out),
            "--manuscript", str(ms),
        ])
        # Clean: no hash drift, no drifted claims (all review stubs)
        assert rc == 0

    def test_audit_detects_hash_drift(self, tmp_path):
        ms = tmp_path / "paper.md"
        ms.write_text("N=50 respondents\n", encoding="utf-8")
        claims_out = tmp_path / "claims.yaml"
        main(["extract", str(ms), "-o", str(claims_out)])
        # Now change the manuscript after extraction
        ms.write_text("N=50 respondents modified\n", encoding="utf-8")
        rc = main([
            "audit", str(claims_out),
            "--manuscript", str(ms),
        ])
        assert rc == 1  # hash drift → NOT OK


# ---------------------------------------------------------------------------
# CLI: quiet flag and HTML output
# ---------------------------------------------------------------------------

class TestCLIFlags:
    def test_quiet_flag_single_line(self, tmp_path, capsys):
        df = pd.DataFrame({"x": [1, 2, 3]})
        (tmp_path / "d.csv").write_text(df.to_csv(index=False))
        claims_file = tmp_path / "claims.yaml"
        doc = {
            "meta": {},
            "claims": [
                {"id": "c1", "value": 3, "comparison": "abs", "tolerance": 0,
                 "binds_to": {"op": "count_rows", "source": "d.csv"}}
            ]
        }
        claims_file.write_text(yaml.safe_dump(doc))
        main(["verify", str(claims_file), "--quiet"])
        out = capsys.readouterr().out.strip()
        assert out.count("\n") == 0  # single line

    def test_html_output_written(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        (tmp_path / "d.csv").write_text(df.to_csv(index=False))
        claims_file = tmp_path / "claims.yaml"
        html_out = tmp_path / "report.html"
        doc = {
            "meta": {},
            "claims": [
                {"id": "c1", "value": 1, "comparison": "abs", "tolerance": 0,
                 "binds_to": {"op": "count_rows", "source": "d.csv"}}
            ]
        }
        claims_file.write_text(yaml.safe_dump(doc))
        main(["verify", str(claims_file), "--html", str(html_out)])
        assert html_out.exists()
        content = html_out.read_text()
        assert "claimpin" in content
        assert "PASS" in content
