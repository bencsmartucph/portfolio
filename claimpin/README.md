# claimpin

Most empirical papers are static prose with typed-in numbers. Those numbers drift. An analyst updates a model, regenerates a table, edits a coefficient into the text — and the previous number stays in an earlier paragraph. Statcheck catches internal inconsistencies in reported test statistics, but only for a narrow family of statistics and only within the document. Dynamic documents (R Markdown, Quarto) help only if the paper was authored that way from the start. claimpin takes a different approach: for each number a paper asserts, you write down *what artifact or recomputation grounds it* in a plain YAML file. `claimpin verify` resolves every binding against the committed artifact and fails loudly on any mismatch. `claimpin audit` checks whether the prose has drifted away from the numbers the claims file was extracted from. The cost is a short curation step; the benefit is that no number can drift silently past a CI run or a pre-submission check.

---

## 5-minute quickstart

**Install** (editable, so the examples resolve correctly):

```
pip install -e .
```

**Run the quickstart:**

```
claimpin verify examples/quickstart/claims.yaml
```

Expected output:

```
claimpin verify — <absolute path to>\examples\quickstart\claims.yaml
  13 passed, 0 failed, 0 errored, 1 skipped (review)
  SKIP  replication_effect_smaller (lines 20, 21): REVIEW: no committed ground truth — No committed artifact for the unpublished replication — skipped loudly until one exists.
  OK
```

The skip is deliberate. The claim `replication_effect_smaller` has `review: true` because no committed artifact exists for the unpublished replication it cites. Review-flagged claims are always reported loudly in the output; they are never silently passed. The exit code is 0 because skipped claims are not failures — only `fail` and `error` make a run unsuccessful.

**Mutation demo** — see what a failure looks like:

1. In `examples/quickstart/paper.md` line 9, change `averages 10.59 pages` to `averages 10.95 pages`.
2. In `examples/quickstart/claims.yaml`, change `value: 10.59` (the `mean_pages` claim) to `value: 10.95`.
3. Re-run `claimpin verify examples/quickstart/claims.yaml`.

You will see exactly one failure (the SKIP line stays, as always):

```
  FAIL  mean_pages (lines 9): claim=10.95 truth=10.5869 |diff|=3.63e-01 tol=0.005
  SKIP  replication_effect_smaller (lines 20, 21): REVIEW: ...
  NOT OK
```

Exit code is 1. Revert both edits to restore the passing state.

(A confession that doubles as a demo: the first draft of this README typed these
output blocks in by hand, and a fresh-eyes review found the truth value and the
diff had drifted from what the tool actually prints. Numbers retyped into static
prose drift — that is the whole pitch.)

---

## Workflow

**1. Extract — bootstrap a skeleton**

```
claimpin extract paper.md -o claims.yaml
```

Scans keyword-bearing lines (lines containing `%`, `β`, `N=`, `mean`, `p <`, and related patterns) and writes a stub claim for every number found. Every stub has `review: true`, so it is skipped loudly until you curate it. Nothing passes silently.

`extract` refuses to overwrite an existing claims file (your curation is the most labour-intensive artifact in this workflow); pass `--force` if you really mean to start over. The keyword heuristic misses numbers whose keyword sits on a neighbouring line — `audit`'s coverage report is the safety net for what `extract` missed.

**2. Curate bindings — the human step**

Open `claims.yaml` and bind each stub to its ground-truth artifact. This step is deliberately not automated. A wrong guessed binding that passes silently is worse than no binding at all. For each claim:

- Set `binds_to` to the op and source that reproduce the number (see the reference below).
- Set `review: false` (or remove the key) once you trust the binding.
- Delete stubs for numbers you choose not to pin (years, footnote references, etc.).

**3. Verify — in CI or before submission**

```
claimpin verify claims.yaml
```

Returns exit 0 if all bound claims pass and all skipped claims are explicit. Returns exit 1 on any `fail` or `error`. Wire this into your CI pipeline or pre-commit hook.

**4. Audit — prose drift and coverage**

```
claimpin audit claims.yaml
```

Checks three things: (a) whether the manuscript file has changed since extraction (hash drift), (b) whether the digits in each claim's `text_snippet` still appear on the stated manuscript lines (per-claim prose drift), (c) how many keyword-bearing numeric lines have no claim covering them (coverage gaps).

Note: running `audit` immediately after `extract` reports `OK` with zero drift — fresh stubs were extracted from the very lines they cite, so nothing can have drifted yet. Drift detection earns its keep later, when the manuscript is edited after curation.

---

## claims.yaml reference

A claims file has two top-level keys: `meta` and `claims`.

### `meta`

| Key | Meaning |
|---|---|
| `manuscript` | Path to the manuscript file (relative to the claims file's directory). Used by `audit`. |
| `manuscript_sha256` | Hash of the manuscript at extraction time. `audit` reports drift if it changes. |
| `ops_module` | Path to a per-project plugin file (relative to the claims file). Loaded by `verify`. |
| `project_root` | Base path for resolving `source` in bindings. Defaults to the claims file's directory. |
| `generated` | Date of extraction (informational). |

### Each claim

| Field | Meaning |
|---|---|
| `id` | Unique slug. Used in report output and as the test identifier. |
| `text_snippet` | The manuscript prose being pinned — the sentence or phrase containing the number. |
| `manuscript_lines` | Line number(s) where the snippet lives. Used by `audit` to detect prose drift. |
| `kind` | Free-form tag for the type of claim: `n`, `coefficient`, `percent`, `p_value`, `correlation`, etc. |
| `value` | The number the manuscript asserts. |
| `comparison` | How to compare claimed value against ground truth: `abs` (within tolerance), `lt` (truth must be less than value), `gt` (truth must be greater than value). Default: `abs`. |
| `tolerance` | Absolute tolerance for `abs` comparison. A tolerance of 0 requires exact integer match (a small float epsilon is added internally to absorb representation noise). |
| `binds_to` | Mapping with `op` plus op-specific parameters that resolve to a ground-truth float. Required unless `review: true`. |
| `review` | If `true`, the claim has no committed ground truth and is skipped loudly. Never silently passed. |
| `notes` | Free-form provenance note. Shown in skip messages. |

### Built-in ops

**Derived (no source file)**

| Op | Parameters | Computes |
|---|---|---|
| `sum` | `terms:` list of bindings | Sum of sub-bindings resolved recursively |
| `pct_reduction` | `claim_a:`, `claim_b:` claim ids | `(1 - value_b / value_a) * 100` as an internal consistency check between two claimed values |

**JSON sources (`source: path/to/file.json`)**

| Op | Parameters | Computes |
|---|---|---|
| `lookup` | `path:` slash-separated key path | Value at that path in the JSON document |
| `z_p` | `coef_path:`, `se_path:` | Two-sided p-value from a z-statistic: `coef / se` |
| `tost_p` | `beta_path:`, `se_path:`, `n_path:`, `sesoi:` | Two one-sided t-tests (TOST) against `|beta| = sesoi`, returns the larger p-value |

**CSV sources (`source: path/to/file.csv`)**

| Op | Parameters | Computes |
|---|---|---|
| `lookup` | `row:` filter dict, `field:` column name | Value of `field` in the single matching row. Omit `row:` only for a one-row file; the op errors unless exactly one row matches |
| `count_rows` | — | Number of rows in the file |
| `count_where` | `where: [col, op, val]` | Count of rows satisfying the condition (`<`, `<=`, `>`, `>=`, `==`) |
| `col_stat` | `field:`, `stat:`, optionally `where:` | Any pandas Series method on a column (e.g. `mean`, `median`, `std`) |
| `nunique` | `col:` | Number of distinct values in a column |
| `notnull_count` | `col:` | Count of non-null values |
| `notnull_pct` | `col:` | Percentage of non-null values |
| `ceiling_pct` | `col:` | Percentage of values equal to the column maximum |
| `cronbach_alpha` | `cols:` list of column names | Cronbach's alpha for an item battery |
| `pearson_r` | `x:`, `y:` column names | Pearson correlation coefficient |
| `pearson_p` | `x:`, `y:` | Two-sided p-value for the Pearson correlation |
| `pearson_r2_pct` | `x:`, `y:` | R² as a percentage (`r² × 100`) |

---

## Plugin guide

For paper-specific computations — group differences on a custom analysis sample, model coefficients from a fitted object, threshold checks — add an `ops.py` file alongside your `claims.yaml` and register it in meta:

```yaml
meta:
  ops_module: ops.py
```

### Custom ops with `@claimpin.op`

A custom op receives `(ctx, **params)` where `ctx` is the project context (use `ctx.load_csv(rel)` and `ctx.load_json(rel)`) and `params` are the binding's keys minus `op`. It must return a float.

```python
import claimpin

@claimpin.op("cohort_gap")
def cohort_gap(ctx, source, outcome, group_col, group_a, group_b):
    """Mean difference in `outcome` between two groups."""
    df = ctx.load_csv(source)
    a = df[df[group_col] == group_a][outcome].mean()
    b = df[df[group_col] == group_b][outcome].mean()
    return float(a - b)
```

In `claims.yaml`:

```yaml
- id: cohort_writing_gap
  value: 0.41
  comparison: abs
  tolerance: 0.005
  binds_to:
    op: cohort_gap
    source: results/survey.csv
    outcome: pages_per_week
    group_col: cohort
    group_a: econ
    group_b: polsci
```

### Custom checks with `@claimpin.check`

A custom check receives `(ctx,)` and signals failure by raising `AssertionError`. Use these for assertions that are not single-number claims: sign expectations, plausibility bands, comparisons of a fitted model against a committed result.

```python
@claimpin.check("coffee_effect_is_negative")
def coffee_effect_is_negative(ctx):
    """Theory says distance hurts output: the coefficient must be negative."""
    results = ctx.load_json("results/toy_results.json")
    coef = results["model1"]["coef_coffee_distance"]
    assert coef < 0, f"coffee-distance coefficient is non-negative ({coef:.4f}) — sign flip"
```

Custom checks are run by `claimpin verify` after all claims. They appear in the report alongside claims.

The full quickstart plugin is at `examples/quickstart/ops.py`.

---

## Output formats

`claimpin verify` prints a colour terminal summary by default. Additional formats:

```
claimpin verify claims.yaml --json report.json    # machine-readable full report
claimpin verify claims.yaml --html report.html    # claim-beside-ground-truth HTML table
claimpin verify claims.yaml --quiet               # one summary line only
```

The exit code is 0 if every bound claim passed and every skipped claim was explicitly flagged; 1 if any claim failed or errored.

---

## Provenance and honest scope

claimpin was extracted from a verification harness built for one real seminar paper in political economy — 99 curated claims and 118 checks on that paper (those numbers describe the source project, not this package's own test suite under `tests/`). The extraction was validated against that harness test for test: claimpin reproduces the original verdict on every check. The built-in ops and the extract/audit heuristics reflect the patterns that came up in that workflow. They are offered as a tested starting point, not a standard. You will almost certainly need custom ops for any real paper; the plugin system is the intended extension point. The year-exclusion heuristic in `extract` (bare four-digit years in citation ranges) is tuned to one genre of writing and will miss some numbers and capture others in different contexts.

---

## Development

```
pip install -e .[dev]
python -m pytest tests -q
```

---

## License

MIT.
