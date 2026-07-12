"""
Headline-r reconcile (2026-05-29 final sprint).

Purpose: lock the §V.D / §V.G headline correlation and the jackknife sub-values
to ONE computation, resolving paper(-0.848) vs notebook(-0.855) vs the script's
internally-inconsistent comments (-0.808 vs -0.802; -0.700 vs -0.717).

This is the BLUPs block from scripts/random_slopes_models.py (lines ~314-393)
copied VERBATIM, plus the minimal df/df3 setup that script uses (lines 31-32, 128).
It does NOT refit Models 1-5, so it cannot perturb any other cited number.

Run: python analysis/reconcile_headline_r_2026-05-29.py
"""
import pandas as pd
import numpy as np
import patsy
import itertools
from statsmodels.regression.mixed_linear_model import MixedLM as _MixedLM
from scipy import stats as _stats
import warnings
warnings.filterwarnings('ignore')

# ── Minimal setup, identical to random_slopes_models.py ──────────────────────
df = pd.read_csv('analysis/sorting_mechanism_master_v2.csv', low_memory=False)
df = df.dropna(subset=['task_z', 'anti_immig_index'])
df3 = df.dropna(subset=['cwed_generosity_z'])

# ── BLUPs block (verbatim from random_slopes_models.py) ──────────────────────
print("\nBLUPs jackknife (§V.D — country grouping, with controls)...")

ctrl_cols_blup = ['agea', 'age_sq', 'female', 'college', 'hinctnta', 'urban']
df_blup = df3[['task_z', 'anti_immig_index', 'cntry', 'cwed_generosity'] + ctrl_cols_blup].dropna()
formula_blup = 'anti_immig_index ~ task_z + ' + ' + '.join(ctrl_cols_blup)
endog_blup, exog_blup = patsy.dmatrices(formula_blup, data=df_blup, return_type='dataframe')
endog_blup = endog_blup.iloc[:, 0]
groups_blup = df_blup['cntry'].astype(str).tolist()
re_blup = patsy.dmatrix('~task_z', data=df_blup, return_type='dataframe')

m_blup = _MixedLM(endog_blup, exog_blup, groups=groups_blup, exog_re=re_blup).fit(
    reml=True, method='lbfgs', disp=False)
fixed_slope = m_blup.params['task_z']
ranef = m_blup.random_effects
cwed_lookup = df_blup.groupby('cntry')['cwed_generosity'].first().to_dict()

blup_data = []
for c, vals in ranef.items():
    blup_slope = fixed_slope + (vals.get('task_z', 0) if hasattr(vals, 'get') else (vals['task_z'] if 'task_z' in vals.index else 0))
    if c in cwed_lookup:
        blup_data.append({'cntry': c, 'blup_slope': float(blup_slope), 'cwed': cwed_lookup[c]})
blup_df = pd.DataFrame(blup_data).sort_values('cntry').reset_index(drop=True)

r_blup_full, p_blup_full = _stats.pearsonr(blup_df['cwed'], blup_df['blup_slope'])
print(f"  HEADLINE full-sample BLUPs correlation: r={r_blup_full:.4f}, p={p_blup_full:.6f}, N={len(blup_df)}")
print(f"  R-squared (variance explained): {r_blup_full**2:.3f}  ({r_blup_full**2*100:.0f}%)")

# Single-country jackknife on BLUPs
single_blup_rows = []
for c in blup_df['cntry']:
    sub = blup_df[blup_df['cntry'] != c]
    r_, p_ = _stats.pearsonr(sub['cwed'], sub['blup_slope'])
    single_blup_rows.append({'excluded': c, 'r': r_, 'p': p_, 'n': len(sub)})
single_blup_df = pd.DataFrame(single_blup_rows).sort_values('r').reset_index(drop=True)

# Two-country jackknife on BLUPs
pair_blup_rows = []
for c1, c2 in itertools.combinations(blup_df['cntry'].tolist(), 2):
    sub = blup_df[~blup_df['cntry'].isin([c1, c2])]
    if len(sub) < 5:
        continue
    r_, p_ = _stats.pearsonr(sub['cwed'], sub['blup_slope'])
    pair_blup_rows.append({'excl_a': c1, 'excl_b': c2, 'r': r_, 'p': p_, 'n': len(sub)})
pair_blup_df = pd.DataFrame(pair_blup_rows).sort_values('r').reset_index(drop=True)

print(f"\n  Single-country jackknife range: r in [{single_blup_df['r'].min():.4f}, {single_blup_df['r'].max():.4f}]")
gb_row = single_blup_df[single_blup_df['excluded'] == 'GB']
no_row = single_blup_df[single_blup_df['excluded'] == 'NO']
if len(gb_row):
    print(f"    Excl GB (UK): r={gb_row.iloc[0]['r']:.4f}, p={gb_row.iloc[0]['p']:.4f}")
if len(no_row):
    print(f"    Excl NO:      r={no_row.iloc[0]['r']:.4f}, p={no_row.iloc[0]['p']:.4f}")

uk_no_pair = pair_blup_df[((pair_blup_df['excl_a'] == 'GB') & (pair_blup_df['excl_b'] == 'NO')) |
                          ((pair_blup_df['excl_a'] == 'NO') & (pair_blup_df['excl_b'] == 'GB'))]
if len(uk_no_pair):
    print(f"    Excl GB+NO:   r={uk_no_pair.iloc[0]['r']:.4f}, p={uk_no_pair.iloc[0]['p']:.4f}")

n_crossing_blup = (pair_blup_df['r'] >= 0).sum()
n_significant_blup = (pair_blup_df['p'] < 0.05).sum()
print(f"\n  Two-country jackknife: {len(pair_blup_df)} pairs")
print(f"    Range: r in [{pair_blup_df['r'].min():.4f}, {pair_blup_df['r'].max():.4f}]")
print(f"    Pairs with r >= 0 (sign flip): {n_crossing_blup} of {len(pair_blup_df)}")
print(f"    Pairs with p < 0.05:           {n_significant_blup} of {len(pair_blup_df)}")

# Denmark slope cross-check (paper line 191 says beta=0.24)
print("\n  Per-country BLUP slopes (paper line 191 claims Denmark beta=0.24):")
print(blup_df.to_string(index=False))
