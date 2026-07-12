"""
Diagnose why the published §V.D r=-0.848 doesn't reproduce on current data.
Tests four hypotheses for what slope methodology produces the headline.
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats as st

df = pd.read_csv('analysis/sorting_mechanism_master_v2.csv', low_memory=False)
df = df.dropna(subset=['task_z', 'anti_immig_index'])
df_c = df.dropna(subset=['cwed_generosity']).copy()
print(f"CWED-available sample: {len(df_c):,} obs, {df_c['cntry'].nunique()} countries")
print(f"Countries: {sorted(df_c['cntry'].unique())}")

# H1: Bivariate per-country OLS (what I just ran)
rows = []
for ctry in sorted(df_c['cntry'].unique()):
    dc = df_c[df_c['cntry'] == ctry][['task_z', 'anti_immig_index', 'cwed_generosity']].dropna()
    if len(dc) < 50:
        continue
    s, _, _, _, se = st.linregress(dc['task_z'], dc['anti_immig_index'])
    rows.append({'cntry': ctry, 'slope_h1': s, 'cwed': dc['cwed_generosity'].iloc[0], 'n': len(dc)})
sl1 = pd.DataFrame(rows)
r1, p1 = st.pearsonr(sl1['cwed'], sl1['slope_h1'])
print(f"\nH1 (bivariate per-country OLS): r = {r1:.3f}, p = {p1:.4f}")

# H2: Per-country OLS WITH controls
ctrl_cols = ['agea', 'age_sq', 'female', 'college', 'hinctnta', 'urban']
rows2 = []
for ctry in sorted(df_c['cntry'].unique()):
    dc = df_c[df_c['cntry'] == ctry][['task_z', 'anti_immig_index', 'cwed_generosity'] + ctrl_cols].dropna()
    if len(dc) < 50:
        continue
    X = sm.add_constant(dc[['task_z'] + ctrl_cols])
    m = sm.OLS(dc['anti_immig_index'], X).fit()
    rows2.append({'cntry': ctry, 'slope_h2': m.params['task_z'],
                  'cwed': dc['cwed_generosity'].iloc[0], 'n': len(dc)})
sl2 = pd.DataFrame(rows2)
r2, p2 = st.pearsonr(sl2['cwed'], sl2['slope_h2'])
print(f"H2 (per-country OLS with controls): r = {r2:.3f}, p = {p2:.4f}")

# H3: BLUPs from random-slopes mixed model
print("H3 (BLUPs from random-slopes MixedLM)... fitting...")
formula = 'anti_immig_index ~ task_z + ' + ' + '.join(ctrl_cols)
df_fit = df_c[['task_z', 'anti_immig_index', 'cntry'] + ctrl_cols].dropna()
import patsy
from statsmodels.regression.mixed_linear_model import MixedLM
endog, exog = patsy.dmatrices(formula, data=df_fit, return_type='dataframe')
endog = endog.iloc[:, 0]
groups = df_fit['cntry'].astype(str).tolist()
exog_re = patsy.dmatrix('~task_z', data=df_fit, return_type='dataframe')
m_rs = MixedLM(endog, exog, groups=groups, exog_re=exog_re).fit(reml=True, method='lbfgs')
fixed_slope = m_rs.params['task_z']
re = m_rs.random_effects  # dict: country → series with intercept and task_z
blup_rows = []
cwed_lookup = df_c.groupby('cntry')['cwed_generosity'].first().to_dict()
for c, vals in re.items():
    blup_slope = fixed_slope + vals['task_z'] if 'task_z' in vals else fixed_slope
    if c in cwed_lookup:
        blup_rows.append({'cntry': c, 'slope_h3': blup_slope, 'cwed': cwed_lookup[c]})
sl3 = pd.DataFrame(blup_rows)
r3, p3 = st.pearsonr(sl3['cwed'], sl3['slope_h3'])
print(f"H3 (BLUPs from MixedLM with controls): r = {r3:.3f}, p = {p3:.4f}")

# H4: Per-country slope from country-wave (treating cntry_wave as unit, then averaging)
print("\nH4 (cntry_wave-level slopes, then averaged within country)...")
rows4 = []
df_c_wave = df_c.copy()
for cw in sorted(df_c_wave['cntry_wave'].dropna().unique()):
    dc = df_c_wave[df_c_wave['cntry_wave'] == cw][['task_z', 'anti_immig_index', 'cwed_generosity', 'cntry']].dropna()
    if len(dc) < 50:
        continue
    s, _, _, _, se = st.linregress(dc['task_z'], dc['anti_immig_index'])
    rows4.append({'cntry_wave': cw, 'cntry': dc['cntry'].iloc[0], 'slope': s,
                  'cwed': dc['cwed_generosity'].iloc[0]})
sl4_wave = pd.DataFrame(rows4)
sl4 = sl4_wave.groupby('cntry').agg(slope_h4=('slope', 'mean'), cwed=('cwed', 'first')).reset_index()
r4, p4 = st.pearsonr(sl4['cwed'], sl4['slope_h4'])
print(f"H4 (cntry_wave slopes averaged within country): r = {r4:.3f}, p = {p4:.4f}")

# Side-by-side per-country slope comparison
print("\nPer-country slope comparison across methods:")
merged = sl1.merge(sl2[['cntry', 'slope_h2']], on='cntry').merge(sl3[['cntry', 'slope_h3']], on='cntry').merge(sl4[['cntry', 'slope_h4']], on='cntry')
print(merged[['cntry', 'cwed', 'slope_h1', 'slope_h2', 'slope_h3', 'slope_h4']].to_string(index=False))
