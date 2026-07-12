# Case study: the seminar paper (golden parity)

claimpin was extracted from a hand-built verification harness
(`Research_Master/verification/`, 118 tests pinning a real political-economy
seminar paper to its artifacts). This case study is the proof that the
extraction lost nothing: the same 99 claims and the same 19 hand-written
checks, expressed in claimpin's format, must reproduce the original suite's
verdict test for test.

This directory only runs on a machine with `Research_Master` checked out as a
sibling of the claimpin repo (paths in `claims.yaml` meta). It is a parity
fixture, not a tutorial — start with `examples/quickstart/` instead.

| File | What it is |
|---|---|
| `claims.yaml` | the paper's 99 claims, ported unchanged (only meta paths adjusted) |
| `ops.py` | the paper-specific plugin: 8 `master_*` binding ops + 19 checks (Model 1 refit, theory sanity) whose names mirror the oracle's test names |
| `parity.py` | reruns the original pytest suite, runs claimpin, maps oracle test → claimpin item bijectively, diffs statuses; exit 0 only on zero diffs |
| `parity_report.md` | the committed result: PARITY HOLDS, zero diffs |
| `report.html` | claimpin's claim-beside-number report for the paper |

Negative control (run it yourself): corrupt any `value:` in `claims.yaml`,
rerun `python parity.py` — exactly one named diff, exit 1. Revert, parity holds.
