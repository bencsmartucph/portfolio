"""Golden parity: claimpin's verdict on the seminar paper must equal the
original verification harness's verdict, test by test.

The oracle is Research_Master/verification/ (read-only): 99 claim tests
(84 reproduction + 15 recompute), 4 Model-1 refit tests, 15 sanity tests,
plus one pytest artifact — test_review_flagged is an empty parametrize
(zero review-flagged claims) that pytest reports as a single SKIP. That skip
maps to "claimpin has zero review-skipped items", not to a claimpin item.

Mapping:
  test_reproduction[<id>] / test_recompute_claim[<id>]  ->  claim item <id>
  test_<name> / test_<name>[<param>]                    ->  check item <name>[<param>]
  test_review_flagged (empty parametrize skip)          ->  zero claimpin skips

Done-condition: ZERO diffs and a verified bijection (every oracle test mapped,
every claimpin item consumed). Exit 0 only then.

Run:
  python parity.py                 # reruns the oracle pytest suite (~60s), then claimpin
  python parity.py --oracle-xml X  # reuse a captured junit XML
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH_MASTER = (HERE / "../../../Research_Master").resolve()

STATUS_EQUIV = {"passed": "pass", "skipped": "skip", "failure": "fail", "error": "error"}


def run_oracle(xml_path: Path) -> None:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    cmd = [
        sys.executable, "-m", "pytest", "verification/tests",
        "-m", "not requires_raw_data", "-q", "-p", "no:cacheprovider",
        f"--junitxml={xml_path}",
    ]
    print(f"[parity] running oracle: {' '.join(cmd[3:])}  (in {RESEARCH_MASTER})")
    proc = subprocess.run(cmd, cwd=RESEARCH_MASTER, env=env, capture_output=True, text=True)
    print(proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-500:])


def parse_oracle(xml_path: Path) -> dict[str, str]:
    """junit XML -> {test_name_with_param: passed|skipped|failure|error}"""
    statuses: dict[str, str] = {}
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        name = case.get("name")
        status = "passed"
        for child in case:
            if child.tag in ("skipped", "failure", "error"):
                status = child.tag
        statuses[name] = status
    return statuses


def run_claimpin() -> dict:
    from claimpin.runner import verify
    return verify(HERE / "claims.yaml")


def map_oracle_to_claimpin(oracle: dict[str, str], result: dict) -> tuple[list, list]:
    items = {it["id"]: it for it in result["items"]}
    consumed: set[str] = set()
    diffs, notes = [], []

    for test_name, oracle_status in sorted(oracle.items()):
        if test_name.startswith("test_review_flagged"):
            # Empty-parametrize artifact: equivalent claimpin state is "no
            # review-skipped items at all".
            n_skips = sum(1 for it in result["items"] if it["status"] == "skip")
            if oracle_status == "skipped" and n_skips == 0:
                notes.append(f"{test_name}: oracle empty-parametrize skip == claimpin zero review-skips (OK)")
            else:
                diffs.append((test_name, oracle_status, f"claimpin has {n_skips} review-skips"))
            continue

        if test_name.startswith(("test_reproduction[", "test_recompute_claim[")):
            cid = test_name.split("[", 1)[1].rstrip("]")
        else:
            cid = test_name.removeprefix("test_")

        item = items.get(cid)
        if item is None:
            diffs.append((test_name, oracle_status, f"NO claimpin item named {cid!r}"))
            continue
        consumed.add(cid)
        want = STATUS_EQUIV[oracle_status]
        if item["status"] != want:
            diffs.append((test_name, oracle_status, f"claimpin {cid}: {item['status']} ({item['message']})"))

    unconsumed = sorted(set(items) - consumed)
    for cid in unconsumed:
        diffs.append((f"(no oracle test)", "-", f"claimpin item {cid!r} maps to no oracle test"))
    return diffs, notes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-xml", default=None)
    parser.add_argument("--report", default=str(HERE / "parity_report.md"))
    args = parser.parse_args()

    if args.oracle_xml:
        xml_path = Path(args.oracle_xml)
    else:
        xml_path = Path(tempfile.mkstemp(suffix="_oracle.xml")[1])
        run_oracle(xml_path)

    oracle = parse_oracle(xml_path)
    result = run_claimpin()
    diffs, notes = map_oracle_to_claimpin(oracle, result)

    s = result["summary"]
    lines = [
        "# Golden parity report — seminar paper",
        "",
        f"- oracle: {len(oracle)} tests "
        f"({sum(1 for v in oracle.values() if v == 'passed')} passed, "
        f"{sum(1 for v in oracle.values() if v == 'skipped')} skipped, "
        f"{sum(1 for v in oracle.values() if v in ('failure', 'error'))} failed/errored)",
        f"- claimpin: {len(result['items'])} items "
        f"({s['pass']} passed, {s['skip']} skipped, {s['fail']} failed, {s['error']} errored)",
        f"- mapping notes: {'; '.join(notes) if notes else 'none'}",
        "",
    ]
    if diffs:
        lines.append(f"## DIFFS ({len(diffs)}) — PARITY FAILED")
        lines += [f"- `{t}` (oracle: {o}) vs {c}" for t, o, c in diffs]
    else:
        lines.append("## DIFFS: none — PARITY HOLDS")
        lines.append("")
        lines.append("Every oracle test maps to exactly one claimpin item with the "
                     "equivalent status, and every claimpin item is consumed by the mapping.")
    report = "\n".join(lines) + "\n"
    Path(args.report).write_text(report, encoding="utf-8")
    print(report)
    return 1 if diffs else 0


if __name__ == "__main__":
    raise SystemExit(main())
