# claimpin — roadmap & field notes

Open limitations and one field test, recorded during development. Not a backlog to burn
down — a record of what the tool can't yet do and where that surfaced.

## Field test — Im, Mayer, Palier & Rovny (2021), ESS automation-risk paper

claimpin was pointed at a second, unseen paper from an on-disk replication package
(402 MB ESS CSV). 13/15 descriptive claims bound and passed; the 2 it couldn't are the
paper's headline multinomial logit (Stata `mlogit`, no committed results file — see
limitation 3). The exercise was circular at the paper level (no PDF, so the "claims"
were computed from the same CSV they were then checked against), so it is **not** kept
as a case study.

One finding is worth keeping. The first binding for mean age treated ESS codes
`[999, 77, 88, 99]` as missing — silently dropping respondents *actually* aged 77/88/99
and biasing mean age down 0.36 years (50.13 vs 50.49). claimpin's tolerance check caught
it (`|diff| 0.36 > tol 0.05`) and forced the fix (replace only the true missing code,
999). A clean example of the silent missing-code error claimpin exists to catch.

## Roadmap / known limitations

1. **Pearson on string-coded columns — DONE (`ed45c98`).** Built-in `pearson_r/_p/_r2_pct`
   now coerce via `pd.to_numeric(errors="coerce")`; survey value-label exports no longer
   raise. Covered by `test_pearson_coerces_string_label_columns`.
2. **No shared sample-filter.** Every custom op re-implements a paper's analytic-sample
   filter. A `sample_filter:` meta key (or a `ctx.sample()` helper) would remove the
   duplication. ~half-day: `schema.py` + `context.py` + a test + one design call (how an
   op opts into filtered vs full data). Build when a real multi-claim paper needs it —
   no consumer yet.
3. **No Stata interop.** Headline coefficients often live only in Stata output, not a
   JSON/CSV results file. Running `.do` files / parsing `.smcl` is large and fragile; the
   cheaper rule is to require a committed results file and mark `review: true` otherwise.
4. **No manuscript file → `extract` unusable.** Fine by design: claimpin pins *your*
   manuscript, not third-party PDFs that ship without one.
5. **String-label survey columns still break `count_where` / `col_stat`.** pearson now
   coerces; the comparison and stat ops still assume numeric/comparable columns. Small
   fix: coerce in-op, add a `coerce: true` binding flag, or document the custom-op route.
