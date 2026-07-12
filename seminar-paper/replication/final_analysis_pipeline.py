"""
Final Analysis Pipeline — Sorting Mechanism Paper
==================================================
Integrates CWED universalism data, constructs radical right vote,
runs full model suite, produces publication-ready outputs.

Author: Claude (2026-03-16)
Depends on: analysis/sorting_mechanism_master.csv (from overnight run)
"""

# --- Config ---
import sys
import io
# Fix Windows cp1252 encoding issues with Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
import traceback
import sys

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

ROOT = Path(os.environ.get('RESEARCH_MASTER_ROOT', '.'))  # set to your Research_Master checkout; restricted data lives under ROOT/data/raw
MASTER_FILE = ROOT / 'analysis' / 'sorting_mechanism_master.csv'
CWED_FILE = ROOT / 'data' / 'raw' / 'CWED' / 'cwed-subset.csv'
CROSSWALK_FILE = ROOT / 'data' / 'raw' / 'langenkamp_2022' / 'ess_populist_crosswalk.csv'
GPS_FILE = ROOT / 'data' / 'raw' / 'baccini_2024' / 'Replication V3' / 'Data' / 'Raw Data' / 'Global_Party_Survey_by_Party_Stata_V1_10_Feb_2020.dta'
FIG_DIR = ROOT / 'outputs' / 'figures'
TAB_DIR = ROOT / 'outputs' / 'tables'
ANALYSIS_DIR = ROOT / 'analysis'

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# --- Publication plot style ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

REGIME_COLORS = {
    'Nordic': '#2166AC',
    'Continental': '#67A9CF',
    'Liberal': '#D6604D',
    'Southern': '#F4A582',
    'Eastern': '#B2ABD2',
}

REGIME_ORDER = ['Nordic', 'Continental', 'Liberal', 'Southern', 'Eastern']

# --- Tracking ---
results = {}  # Store model results for report
validation_log = {}

# ============================================================
# SECTION 1: Load Master Dataset
# ============================================================
print('\n' + '='*60)
print('  SECTION 1: Load Master Dataset')
print('='*60)

master = pd.read_csv(MASTER_FILE)
pre_n = len(master)
print(f'Master dataset: {master.shape[0]:,} rows x {master.shape[1]} columns')
print(f'Regimes: {master["welfare_regime"].value_counts().to_dict()}')
print(f'Waves: {sorted(master["essround"].unique())}')
print(f'Countries: {sorted(master["cntry"].unique())}')

# Standardize task (RTI) for comparable coefficients
master['task_z'] = (master['task'] - master['task'].mean()) / master['task'].std()
print(f'RTI (task): mean={master["task"].mean():.3f}, sd={master["task"].std():.3f}')
print(f'Anti-immig index: mean={master["anti_immig_index"].mean():.3f}, sd={master["anti_immig_index"].std():.3f}')

validation_log['Load master'] = {
    'rows': pre_n, 'cols': master.shape[1], 'status': 'PASSED'
}

# ============================================================
# SECTION 2: Integrate CWED Universalism Data
# ============================================================
print('\n' + '='*60)
print('  SECTION 2: Integrate CWED Universalism Data')
print('='*60)

# --- 2a: Load and profile CWED ---
cwed = pd.read_csv(CWED_FILE)
print(f'CWED raw: {cwed.shape[0]} rows x {cwed.shape[1]} columns')
print(f'Countries: {cwed["COUNTRY ABBREV"].nunique()}')
print(f'Year range: {cwed["YEAR"].min()}-{cwed["YEAR"].max()}')

# Key variables for welfare generosity:
# TOTGEN: Total generosity score (composite of UE + sickness + pension)
# UEGEN: Unemployment insurance generosity
# SKGEN: Sickness insurance generosity
# PGEN: Pension generosity
# UEWAIT: Waiting days for unemployment benefits (conditionality proxy)
# UECOV: Unemployment insurance coverage rate
# UEQUAL: Qualifying period for unemployment benefits (weeks)

# --- 2b: Map CWED 3-letter country codes to ESS 2-letter codes ---
# CWED uses ISO-3 (AUT, DEU, etc.), ESS uses ISO-2 (AT, DE, etc.)
iso3_to_iso2 = {
    'AUS': 'AU', 'AUT': 'AT', 'BEL': 'BE', 'BGR': 'BG', 'CAN': 'CA',
    'CZE': 'CZ', 'DNK': 'DK', 'EST': 'EE', 'FIN': 'FI', 'FRA': 'FR',
    'DEU': 'DE', 'GRC': 'GR', 'HUN': 'HU', 'IRL': 'IE', 'ITA': 'IT',
    'JPN': 'JP', 'KOR': 'KR', 'LVA': 'LV', 'LTU': 'LT', 'NLD': 'NL',
    'NZL': 'NZ', 'NOR': 'NO', 'POL': 'PL', 'PRT': 'PT', 'ROU': 'RO',
    'SVK': 'SK', 'SVN': 'SI', 'ESP': 'ES', 'SWE': 'SE', 'CHE': 'CH',
    'TWN': 'TW', 'GBR': 'GB', 'USA': 'US',
}
cwed['cntry'] = cwed['COUNTRY ABBREV'].map(iso3_to_iso2)
unmapped = cwed[cwed['cntry'].isna()]['COUNTRY ABBREV'].unique()
if len(unmapped) > 0:
    print(f'WARNING: Unmapped CWED countries: {unmapped}')
cwed = cwed.dropna(subset=['cntry'])

# --- 2c: Construct welfare indicators ---
# CWED ends in 2011, ESS fieldwork starts 2012. Use most recent available year.
# Strategy: for each country, take the mean of 2005-2011 (stable period) to
# reduce year-to-year noise. This gives a time-invariant country characteristic.

cwed_recent = cwed[cwed['YEAR'] >= 2005].copy()
print(f'\nCWED 2005-2011: {len(cwed_recent)} country-years')

# Convert all welfare vars to numeric (TOTGEN stored as string in CSV)
welfare_vars = ['TOTGEN', 'UEGEN', 'SKGEN', 'PGEN', 'UEWAIT', 'UECOV', 'UEQUAL']
for v in welfare_vars:
    cwed_recent[v] = pd.to_numeric(cwed_recent[v], errors='coerce')

# Aggregate to country level (mean over 2005-2011)
cwed_country = cwed_recent.groupby('cntry')[welfare_vars].mean().reset_index()

print(f'\nCWED country-level indicators (mean 2005-2011):')
print(cwed_country[['cntry', 'TOTGEN', 'UEGEN', 'SKGEN', 'UECOV']].to_string(index=False))

# Primary indicator: TOTGEN (total generosity = decommodification)
# Secondary: UEGEN (unemployment generosity — most relevant for automation exposure)
# Conditionality proxy: UEWAIT (waiting days) + UEQUAL (qualifying weeks)

# Construct conditionality index (higher = more conditional/restrictive)
# Normalize UEWAIT and UEQUAL to 0-1 and average
for var in ['UEWAIT', 'UEQUAL']:
    vmin = cwed_country[var].min()
    vmax = cwed_country[var].max()
    if vmax > vmin:
        cwed_country[f'{var}_norm'] = (cwed_country[var] - vmin) / (vmax - vmin)
    else:
        cwed_country[f'{var}_norm'] = 0

cwed_country['conditionality'] = (
    cwed_country['UEWAIT_norm'] + cwed_country['UEQUAL_norm']
) / 2

# --- 2d: Merge onto master ---
ess_countries = set(master['cntry'].unique())
cwed_countries = set(cwed_country['cntry'].unique())
overlap = ess_countries & cwed_countries
missing = ess_countries - cwed_countries
print(f'\nESS countries: {len(ess_countries)}')
print(f'CWED countries: {len(cwed_countries)}')
print(f'Overlap: {len(overlap)} countries: {sorted(overlap)}')
print(f'ESS countries missing from CWED: {sorted(missing)}')

pre_merge_n = len(master)
master = master.merge(
    cwed_country[['cntry', 'TOTGEN', 'UEGEN', 'SKGEN', 'UECOV', 'conditionality']],
    on='cntry', how='left'
)
post_merge_n = len(master)

assert pre_merge_n == post_merge_n, f'Row inflation during CWED merge! {pre_merge_n} -> {post_merge_n}'

cwed_match_rate = master['TOTGEN'].notna().mean()
print(f'\nCWED merge: {pre_merge_n} -> {post_merge_n} rows (no inflation)')
print(f'CWED match rate: {cwed_match_rate:.1%} ({master["TOTGEN"].notna().sum():,} obs with CWED data)')

# Rename for clarity
master = master.rename(columns={
    'TOTGEN': 'cwed_generosity',
    'UEGEN': 'cwed_ue_generosity',
    'SKGEN': 'cwed_sk_generosity',
    'UECOV': 'cwed_ue_coverage',
})

# Standardize CWED indicators for models
for var in ['cwed_generosity', 'cwed_ue_generosity', 'cwed_sk_generosity', 'conditionality']:
    master[f'{var}_z'] = (master[var] - master[var].mean()) / master[var].std()

validation_log['CWED merge'] = {
    'pre_rows': pre_merge_n, 'post_rows': post_merge_n,
    'match_rate': f'{cwed_match_rate:.1%}',
    'countries_matched': len(overlap),
    'status': 'PASSED'
}

print(f'\nCWED generosity by regime:')
print(master.groupby('welfare_regime')['cwed_generosity'].mean().reindex(REGIME_ORDER).to_string())

# ============================================================
# SECTION 3: Construct Radical Right Party Vote
# ============================================================
print('\n' + '='*60)
print('  SECTION 3: Construct Radical Right Party Vote')
print('='*60)

# Manual classification of major radical right parties in ESS countries
# Conservative list — only parties universally classified as radical right
# Sources: Mudde (2019), Rooduijn et al. PopuList, CHES expert surveys
RADICAL_RIGHT_PARTIES = {
    # Format: (cntry, party_name_substring): True
    # We'll match on partyfacts_name or party name substrings
    'AT': ['FPÖ', 'BZÖ', 'FPO', 'BZO'],
    'BE': ['VB', 'Vlaams Belang', 'Vlaams Blok'],
    'BG': ['Ataka', 'VMRO', 'Volya'],
    'CH': ['SVP', 'UDC'],
    'CZ': ['SPD', 'Úsvit'],
    'DE': ['AfD', 'NPD'],
    'DK': ['DF', 'Dansk Folkeparti'],
    'EE': ['EKRE'],
    'ES': ['VOX', 'Vox'],
    'FI': ['PS', 'Perussuomalaiset', 'True Finns'],
    'FR': ['FN', 'RN', 'Front National', 'Rassemblement National'],
    'GB': ['UKIP', 'BNP', 'UK Independence'],
    'GR': ['XA', 'Golden Dawn', 'LAOS', 'Chrysi Avgi'],
    'HU': ['Jobbik', 'Mi Hazánk'],
    'IE': [],  # No significant radical right party
    'IT': ['Lega', 'LN', 'FdI', 'Fratelli'],
    'LT': ['TT'],
    'NL': ['PVV', 'FvD', 'Forum'],
    'NO': ['FrP', 'Fremskrittspartiet'],
    'PL': ['Konfederacja', 'Kukiz'],
    'PT': ['Chega'],
    'SE': ['SD', 'Sverigedemokraterna'],
    'SI': ['SDS', 'SNS'],
    'SK': ['SNS', 'ĽSNS', 'Kotleba'],
}

try:
    # Load Langenkamp crosswalk
    crosswalk = pd.read_csv(CROSSWALK_FILE, sep=';', encoding='latin-1')
    print(f'Langenkamp crosswalk: {crosswalk.shape[0]} rows')
    print(f'Columns: {list(crosswalk.columns)}')

    # Flag radical right parties in the crosswalk
    import re
    def is_radical_right(row):
        """Check if a party matches our radical right classification."""
        cntry = row['cntry']
        party_name = str(row.get('party', ''))
        pf_name = str(row.get('partyfacts_name', ''))

        if cntry not in RADICAL_RIGHT_PARTIES:
            return False

        rr_keywords = RADICAL_RIGHT_PARTIES[cntry]
        for kw in rr_keywords:
            # For short keywords (<=3 chars), require word boundary match
            if len(kw) <= 3:
                pattern = r'\b' + re.escape(kw) + r'\b'
                if re.search(pattern, party_name, re.IGNORECASE) or re.search(pattern, pf_name, re.IGNORECASE):
                    return True
            else:
                if kw.lower() in party_name.lower() or kw.lower() in pf_name.lower():
                    return True
        return False

    crosswalk['radical_right'] = crosswalk.apply(is_radical_right, axis=1)
    rr_parties = crosswalk[crosswalk['radical_right']]
    print(f'\nRadical right parties identified: {len(rr_parties)}')
    print(rr_parties[['cntry', 'essround', 'party', 'ess_id']].to_string())

    # Now we need to match crosswalk to ESS party vote data
    # The crosswalk has: cntry, essround, variable (e.g. 'prtvtde'), ess_id (numeric code)
    # The master dataset doesn't have individual party vote columns — those are in the raw ESS.
    # We need to go back to the raw ESS to get party vote data.

    # Check if master has any party vote info
    prtvt_cols = [c for c in master.columns if c.startswith('prtvt')]
    print(f'\nParty vote columns in master: {prtvt_cols}')

    if len(prtvt_cols) == 0:
        print('\nWARNING: Master dataset has no party vote columns.')
        print('Radical right vote construction requires re-loading raw ESS data.')
        print('This is a significant merge — flagging for manual completion.')

        # Attempt to load from raw ESS CSVs
        ESS_DIR = ROOT / 'data' / 'raw' / 'ESS_csv'
        if ESS_DIR.exists():
            print('\nAttempting to extract party vote data from raw ESS CSVs...')
            vote_dfs = []
            for fpath in sorted(ESS_DIR.glob('*/ESS*.csv')):
                header = pd.read_csv(fpath, nrows=0, encoding='latin-1').columns.tolist()
                prtvt_cols_raw = [c for c in header if c.startswith('prtvt')]
                id_cols = ['essround', 'cntry', 'idno']
                cols_to_load = [c for c in id_cols + prtvt_cols_raw if c in header]
                if prtvt_cols_raw:
                    df_vote = pd.read_csv(fpath, usecols=cols_to_load, encoding='latin-1',
                                          low_memory=False)
                    vote_dfs.append(df_vote)
                    print(f'  {fpath.name}: {len(prtvt_cols_raw)} party vote vars')

            if vote_dfs:
                # Stack and melt party vote columns to long format
                all_votes = pd.concat(vote_dfs, ignore_index=True)
                print(f'\nRaw vote data: {len(all_votes):,} rows')

                # Melt party vote columns to long format
                id_cols = ['essround', 'cntry', 'idno']
                prtvt_all = [c for c in all_votes.columns if c.startswith('prtvt')]
                vote_long = all_votes.melt(
                    id_vars=id_cols,
                    value_vars=prtvt_all,
                    var_name='vote_var',
                    value_name='party_code'
                )
                # Drop rows where party_code is NaN (didn't vote in that country's var)
                vote_long = vote_long.dropna(subset=['party_code'])
                # ESS uses negative codes for missing/refusal — keep only positive
                vote_long['party_code'] = pd.to_numeric(vote_long['party_code'], errors='coerce')
                vote_long = vote_long[vote_long['party_code'] > 0]
                print(f'Vote data (long, valid responses): {len(vote_long):,} rows')

                # Merge with crosswalk to classify
                # crosswalk has: cntry, essround, variable, ess_id, radical_right
                vote_classified = vote_long.merge(
                    crosswalk[['cntry', 'essround', 'variable', 'ess_id', 'radical_right']],
                    left_on=['cntry', 'essround', 'vote_var', 'party_code'],
                    right_on=['cntry', 'essround', 'variable', 'ess_id'],
                    how='left'
                )

                # Anyone who voted but isn't matched to a radical right party = 0
                vote_classified['radical_right'] = vote_classified['radical_right'].fillna(False)

                # Collapse to person level: did they vote radical right?
                person_rr = vote_classified.groupby(['essround', 'cntry', 'idno'])['radical_right'].max().reset_index()
                person_rr['radical_right_vote'] = person_rr['radical_right'].astype(int)
                person_rr = person_rr.drop(columns=['radical_right'])

                print(f'\nPerson-level radical right vote: {len(person_rr):,} voters')
                print(f'Radical right vote rate: {person_rr["radical_right_vote"].mean():.1%}')

                # Merge onto master
                pre_n = len(master)
                master = master.merge(
                    person_rr[['essround', 'cntry', 'idno', 'radical_right_vote']],
                    on=['essround', 'cntry', 'idno'],
                    how='left'
                )
                post_n = len(master)
                assert pre_n == post_n, f'Row inflation! {pre_n} -> {post_n}'

                rr_valid = master['radical_right_vote'].notna().sum()
                rr_rate = master.loc[master['radical_right_vote'].notna(), 'radical_right_vote'].mean()
                print(f'\nRadical right vote merged onto master: {rr_valid:,} obs with data ({rr_valid/len(master):.1%})')
                print(f'Overall radical right vote rate: {rr_rate:.1%}')

                # Sanity check by country
                print('\nRadical right vote share by country:')
                rr_by_country = (master[master['radical_right_vote'].notna()]
                                .groupby('cntry')['radical_right_vote']
                                .agg(['mean', 'sum', 'count']))
                rr_by_country.columns = ['vote_share', 'n_rr', 'n_voters']
                rr_by_country = rr_by_country.sort_values('vote_share', ascending=False)
                print(rr_by_country.head(20).to_string())

                validation_log['Radical right vote'] = {
                    'n_voters': rr_valid,
                    'vote_rate': f'{rr_rate:.1%}',
                    'status': 'PASSED'
                }
        else:
            print('ESS_csv directory not found. Skipping radical right vote construction.')
            master['radical_right_vote'] = np.nan
            validation_log['Radical right vote'] = {'status': 'SKIPPED - no raw ESS data'}
    else:
        print('Party vote columns found in master — unexpected. Investigating...')

except Exception as e:
    print(f'ERROR in radical right vote construction: {e}')
    traceback.print_exc()
    master['radical_right_vote'] = np.nan
    validation_log['Radical right vote'] = {'status': f'FAILED: {e}'}

# ============================================================
# SECTION 4: Final Model Suite
# ============================================================
print('\n' + '='*60)
print('  SECTION 4: Final Model Suite')
print('='*60)

# --- Prepare analysis sample ---
# Core controls for all models
controls = ['agea', 'age_sq', 'female', 'college', 'hinctnta', 'urban']
model_vars_base = ['anti_immig_index', 'task_z', 'welfare_regime', 'cntry_wave'] + controls

# Analysis sample: complete cases on base vars
analysis = master.dropna(subset=model_vars_base).copy()
print(f'Analysis sample (complete cases): {len(analysis):,} / {len(master):,} ({len(analysis)/len(master):.1%})')
print(f'Regimes in sample: {analysis["welfare_regime"].value_counts().to_dict()}')

# Restrict to 5 main regimes (exclude "Other")
analysis = analysis[analysis['welfare_regime'].isin(REGIME_ORDER)].copy()
print(f'After excluding Other: {len(analysis):,}')

# Create regime dummies for interactions
analysis['regime_liberal'] = (analysis['welfare_regime'] == 'Liberal').astype(int)
analysis['regime_continental'] = (analysis['welfare_regime'] == 'Continental').astype(int)
analysis['regime_southern'] = (analysis['welfare_regime'] == 'Southern').astype(int)
analysis['regime_eastern'] = (analysis['welfare_regime'] == 'Eastern').astype(int)
# Nordic is reference

# Non-college indicator for triple interaction
analysis['non_college'] = (1 - analysis['college']).astype(int)

# --- Model 1: Baseline (RTI only) ---
print('\n--- Model 1: Baseline ---')
try:
    formula1 = 'anti_immig_index ~ task_z + agea + age_sq + female + college + hinctnta + urban + C(cntry_wave)'
    m1 = smf.ols(formula1, data=analysis).fit(cov_type='cluster', cov_kwds={'groups': analysis['cntry_wave']})
    print(f'RTI coefficient: {m1.params["task_z"]:.4f} (SE={m1.bse["task_z"]:.4f}, p={m1.pvalues["task_z"]:.4f})')
    print(f'N={m1.nobs:.0f}, R²={m1.rsquared:.4f}')
    results['model1'] = {
        'coef_rti': m1.params['task_z'],
        'se_rti': m1.bse['task_z'],
        'p_rti': m1.pvalues['task_z'],
        'n': int(m1.nobs),
        'r2': m1.rsquared,
    }
except Exception as e:
    print(f'Model 1 failed: {e}')
    traceback.print_exc()

# --- Model 2: Regime interaction (Mixed model) ---
print('\n--- Model 2: Regime interaction (Mixed) ---')
try:
    formula2 = ('anti_immig_index ~ task_z * C(welfare_regime, Treatment(reference="Nordic"))'
                ' + agea + age_sq + female + college + hinctnta + urban')
    m2 = smf.mixedlm(formula2, data=analysis, groups=analysis['cntry_wave']).fit(reml=True)
    print(m2.summary().tables[1].to_string())

    # Extract key interaction coefficients
    m2_params = {}
    for param in m2.params.index:
        if 'task_z' in param:
            m2_params[param] = {
                'coef': m2.params[param],
                'se': m2.bse[param],
                'p': m2.pvalues[param],
            }
    results['model2'] = {
        'params': m2_params,
        'n': int(m2.nobs),
        'converged': m2.converged,
    }

    # Print the key interaction
    lib_key = [k for k in m2.params.index if 'Liberal' in k and 'task_z' in k]
    if lib_key:
        k = lib_key[0]
        print(f'\n*** RTI x Liberal: {m2.params[k]:.4f} (SE={m2.bse[k]:.4f}, p={m2.pvalues[k]:.4f}) ***')
except Exception as e:
    print(f'Model 2 failed: {e}')
    traceback.print_exc()

# --- Model 3: CWED interaction (Mixed model) ---
print('\n--- Model 3: CWED generosity interaction (Mixed) ---')
try:
    # Subset to countries with CWED data
    analysis_cwed = analysis.dropna(subset=['cwed_generosity_z']).copy()
    print(f'CWED analysis sample: {len(analysis_cwed):,} obs ({len(analysis_cwed)/len(analysis):.1%} of main sample)')

    formula3 = ('anti_immig_index ~ task_z * cwed_generosity_z'
                ' + agea + age_sq + female + college + hinctnta + urban')
    m3 = smf.mixedlm(formula3, data=analysis_cwed, groups=analysis_cwed['cntry_wave']).fit(reml=True)
    print(m3.summary().tables[1].to_string())

    results['model3'] = {
        'coef_rti': m3.params['task_z'],
        'se_rti': m3.bse['task_z'],
        'coef_cwed': m3.params['cwed_generosity_z'],
        'se_cwed': m3.bse['cwed_generosity_z'],
        'coef_interaction': m3.params['task_z:cwed_generosity_z'],
        'se_interaction': m3.bse['task_z:cwed_generosity_z'],
        'p_interaction': m3.pvalues['task_z:cwed_generosity_z'],
        'n': int(m3.nobs),
        'converged': m3.converged,
    }
    print(f'\n*** RTI x CWED generosity: {m3.params["task_z:cwed_generosity_z"]:.4f}'
          f' (SE={m3.bse["task_z:cwed_generosity_z"]:.4f},'
          f' p={m3.pvalues["task_z:cwed_generosity_z"]:.4f}) ***')
    print('Theory predicts NEGATIVE interaction: more generous welfare -> weaker RTI->exclusion link')
except Exception as e:
    print(f'Model 3 failed: {e}')
    traceback.print_exc()

# --- Model 3b: CWED UE generosity interaction ---
print('\n--- Model 3b: CWED UE generosity interaction ---')
try:
    formula3b = ('anti_immig_index ~ task_z * cwed_ue_generosity_z'
                 ' + agea + age_sq + female + college + hinctnta + urban')
    m3b = smf.mixedlm(formula3b, data=analysis_cwed, groups=analysis_cwed['cntry_wave']).fit(reml=True)

    results['model3b'] = {
        'coef_interaction': m3b.params['task_z:cwed_ue_generosity_z'],
        'se_interaction': m3b.bse['task_z:cwed_ue_generosity_z'],
        'p_interaction': m3b.pvalues['task_z:cwed_ue_generosity_z'],
        'n': int(m3b.nobs),
    }
    print(f'RTI x UE generosity: {m3b.params["task_z:cwed_ue_generosity_z"]:.4f}'
          f' (p={m3b.pvalues["task_z:cwed_ue_generosity_z"]:.4f})')
except Exception as e:
    print(f'Model 3b failed: {e}')

# --- Model 3c: Conditionality interaction ---
print('\n--- Model 3c: Conditionality interaction ---')
try:
    formula3c = ('anti_immig_index ~ task_z * conditionality_z'
                 ' + agea + age_sq + female + college + hinctnta + urban')
    m3c = smf.mixedlm(formula3c, data=analysis_cwed, groups=analysis_cwed['cntry_wave']).fit(reml=True)

    results['model3c'] = {
        'coef_interaction': m3c.params['task_z:conditionality_z'],
        'se_interaction': m3c.bse['task_z:conditionality_z'],
        'p_interaction': m3c.pvalues['task_z:conditionality_z'],
        'n': int(m3c.nobs),
    }
    print(f'RTI x Conditionality: {m3c.params["task_z:conditionality_z"]:.4f}'
          f' (p={m3c.pvalues["task_z:conditionality_z"]:.4f})')
    print('Theory predicts POSITIVE interaction: more conditional welfare -> stronger RTI->exclusion link')
except Exception as e:
    print(f'Model 3c failed: {e}')

# --- Model 4: Education triple interaction ---
print('\n--- Model 4: Education triple interaction ---')
try:
    formula4 = ('anti_immig_index ~ task_z * C(welfare_regime, Treatment(reference="Nordic")) * non_college'
                ' + agea + age_sq + female + hinctnta + urban')
    m4 = smf.mixedlm(formula4, data=analysis, groups=analysis['cntry_wave']).fit(reml=True)
    print(m4.summary().tables[1].to_string())
    results['model4'] = {
        'params': {k: {'coef': m4.params[k], 'se': m4.bse[k], 'p': m4.pvalues[k]}
                   for k in m4.params.index if 'task_z' in k},
        'n': int(m4.nobs),
        'converged': m4.converged,
    }
except Exception as e:
    print(f'Model 4 failed: {e}')
    traceback.print_exc()

# --- Model 5: Redistribution DV ---
print('\n--- Model 5: Redistribution DV ---')
try:
    analysis_redist = analysis.dropna(subset=['redist_support']).copy()
    formula5 = ('redist_support ~ task_z * C(welfare_regime, Treatment(reference="Nordic"))'
                ' + agea + age_sq + female + college + hinctnta + urban')
    m5 = smf.mixedlm(formula5, data=analysis_redist, groups=analysis_redist['cntry_wave']).fit(reml=True)

    m5_params = {}
    for param in m5.params.index:
        if 'task_z' in param:
            m5_params[param] = {
                'coef': m5.params[param],
                'se': m5.bse[param],
                'p': m5.pvalues[param],
            }
    results['model5'] = {'params': m5_params, 'n': int(m5.nobs)}
    print(f'N={m5.nobs:.0f}')
    for k, v in m5_params.items():
        print(f'  {k}: {v["coef"]:.4f} (p={v["p"]:.4f})')
except Exception as e:
    print(f'Model 5 failed: {e}')

# --- Model 6: Radical Right Vote (Logistic) ---
print('\n--- Model 6: Radical Right Vote (Logistic) ---')
try:
    analysis_rr = analysis.dropna(subset=['radical_right_vote']).copy()
    if len(analysis_rr) > 100 and analysis_rr['radical_right_vote'].sum() > 10:
        formula6 = ('radical_right_vote ~ task_z * C(welfare_regime, Treatment(reference="Nordic"))'
                    ' + agea + age_sq + female + college + hinctnta + urban')
        # Use GEE with logit link for clustered binary outcome
        m6 = smf.logit(formula6, data=analysis_rr).fit(
            cov_type='cluster', cov_kwds={'groups': analysis_rr['cntry_wave']},
            maxiter=100, disp=0)

        m6_params = {}
        for param in m6.params.index:
            if 'task_z' in param:
                m6_params[param] = {
                    'coef': m6.params[param],
                    'se': m6.bse[param],
                    'p': m6.pvalues[param],
                }
        results['model6'] = {'params': m6_params, 'n': int(m6.nobs)}
        print(f'N={m6.nobs:.0f}')
        for k, v in m6_params.items():
            print(f'  {k}: {v["coef"]:.4f} (p={v["p"]:.4f})')
    else:
        print(f'Insufficient data for logistic model: {len(analysis_rr)} obs, {analysis_rr["radical_right_vote"].sum()} RR votes')
        results['model6'] = {'status': 'SKIPPED - insufficient data'}
except Exception as e:
    print(f'Model 6 failed: {e}')
    traceback.print_exc()
    results['model6'] = {'status': f'FAILED: {e}'}

# --- Model 7: Deservingness mediation (Wave 8 only) ---
print('\n--- Model 7: Deservingness mediation (Wave 8 only) ---')
try:
    analysis_w8 = analysis[
        (analysis['essround'] == 8) &
        analysis['narrow_deserving'].notna()
    ].copy()
    print(f'Wave 8 sample with deservingness: {len(analysis_w8):,}')

    if len(analysis_w8) > 100:
        # Step 1: RTI -> deservingness
        formula7a = ('narrow_deserving ~ task_z + agea + age_sq + female + college + hinctnta + urban')
        m7a = smf.mixedlm(formula7a, data=analysis_w8, groups=analysis_w8['cntry_wave']).fit(reml=True)

        # Step 2: Deservingness -> anti-immigration (controlling for RTI)
        formula7b = ('anti_immig_index ~ task_z + narrow_deserving + agea + age_sq + female + college + hinctnta + urban')
        m7b = smf.mixedlm(formula7b, data=analysis_w8, groups=analysis_w8['cntry_wave']).fit(reml=True)

        results['model7'] = {
            'step1_rti_deserving': {
                'coef': m7a.params['task_z'],
                'se': m7a.bse['task_z'],
                'p': m7a.pvalues['task_z'],
            },
            'step2_deserving_antiimmig': {
                'coef': m7b.params['narrow_deserving'],
                'se': m7b.bse['narrow_deserving'],
                'p': m7b.pvalues['narrow_deserving'],
            },
            'step2_rti_direct': {
                'coef': m7b.params['task_z'],
                'se': m7b.bse['task_z'],
                'p': m7b.pvalues['task_z'],
            },
            'n': int(m7a.nobs),
        }
        print(f'Step 1 - RTI -> deservingness: {m7a.params["task_z"]:.4f} (p={m7a.pvalues["task_z"]:.4f})')
        print(f'Step 2 - deservingness -> anti-immig: {m7b.params["narrow_deserving"]:.4f} (p={m7b.pvalues["narrow_deserving"]:.4f})')
        print(f'Step 2 - RTI direct: {m7b.params["task_z"]:.4f} (p={m7b.pvalues["task_z"]:.4f})')
    else:
        print('Insufficient wave 8 data')
        results['model7'] = {'status': 'SKIPPED'}
except Exception as e:
    print(f'Model 7 failed: {e}')
    results['model7'] = {'status': f'FAILED: {e}'}


# ============================================================
# SECTION 5: Robustness Checks
# ============================================================
print('\n' + '='*60)
print('  SECTION 5: Robustness Checks')
print('='*60)

robustness = {}

# Base specification: replicate Model 2's RTI x Liberal interaction
# Use OLS with country-wave FE for faster estimation across many specs

def run_robustness(label, data, formula_extra='', extra_controls=''):
    """Run a robustness variant and extract RTI x Liberal coefficient."""
    try:
        base = ('anti_immig_index ~ task_z * C(welfare_regime, Treatment(reference="Nordic"))'
                ' + agea + age_sq + female + college')
        if 'hinctnta' not in formula_extra and 'no_income' not in label.lower():
            base += ' + hinctnta'
        if 'urban' not in formula_extra:
            base += ' + urban'
        base += extra_controls + formula_extra
        base += ' + C(cntry_wave)'

        m = smf.ols(base, data=data).fit(cov_type='cluster',
                                          cov_kwds={'groups': data['cntry_wave']})

        lib_key = [k for k in m.params.index if 'Liberal' in k and 'task_z' in k]
        if lib_key:
            k = lib_key[0]
            result = {
                'coef': m.params[k],
                'se': m.bse[k],
                'p': m.pvalues[k],
                'ci_lo': m.conf_int().loc[k, 0],
                'ci_hi': m.conf_int().loc[k, 1],
                'n': int(m.nobs),
            }
            print(f'  {label}: b={result["coef"]:.4f} (SE={result["se"]:.4f}, p={result["p"]:.4f}), N={result["n"]:,}')
            return result
        else:
            print(f'  {label}: Liberal interaction not found in params')
            return None
    except Exception as e:
        print(f'  {label}: FAILED - {e}')
        return None

# R1: Main specification (replication)
print('\nR0: Main specification')
robustness['Main'] = run_robustness('Main', analysis)

# R1: Country FE (already using cntry_wave FE, this is equivalent)
# Skip — OLS with C(cntry_wave) IS country-wave FE

# R2: Exclude Eastern European countries
print('\nR2: Exclude Eastern')
analysis_no_east = analysis[analysis['welfare_regime'] != 'Eastern'].copy()
robustness['Excl. Eastern'] = run_robustness('Excl. Eastern', analysis_no_east)

# R3: Exclude Other (Israel, Russia) — already excluded in our sample
# But let's verify
other_count = master[master['welfare_regime'] == 'Other'].shape[0] if 'Other' in master['welfare_regime'].values else 0
print(f'\nR3: Other obs already excluded from analysis sample ({other_count:,} in full master)')

# R4: Subjective insecurity instead of RTI
print('\nR4: Subjective insecurity (if available)')
if 'emplno' in master.columns:
    # emplno = probability of becoming unemployed in next 12 months
    analysis_subj = analysis.dropna(subset=['emplno']).copy() if 'emplno' in analysis.columns else pd.DataFrame()
    if len(analysis_subj) > 1000:
        analysis_subj['emplno_z'] = (analysis_subj['emplno'] - analysis_subj['emplno'].mean()) / analysis_subj['emplno'].std()
        # Replace task_z with emplno_z in formula
        try:
            formula_r4 = ('anti_immig_index ~ emplno_z * C(welfare_regime, Treatment(reference="Nordic"))'
                          ' + agea + age_sq + female + college + hinctnta + urban + C(cntry_wave)')
            m_r4 = smf.ols(formula_r4, data=analysis_subj).fit(
                cov_type='cluster', cov_kwds={'groups': analysis_subj['cntry_wave']})
            lib_key = [k for k in m_r4.params.index if 'Liberal' in k and 'emplno_z' in k]
            if lib_key:
                k = lib_key[0]
                robustness['Subj. insecurity'] = {
                    'coef': m_r4.params[k], 'se': m_r4.bse[k], 'p': m_r4.pvalues[k],
                    'ci_lo': m_r4.conf_int().loc[k, 0], 'ci_hi': m_r4.conf_int().loc[k, 1],
                    'n': int(m_r4.nobs),
                }
                print(f'  Subj. insecurity x Liberal: {m_r4.params[k]:.4f} (p={m_r4.pvalues[k]:.4f})')
        except Exception as e:
            print(f'  R4 failed: {e}')
    else:
        print(f'  Insufficient data for subjective insecurity: {len(analysis_subj)} obs')
else:
    print('  emplno not in dataset — skipping')

# R5: Without income control
print('\nR5: Without income control')
robustness['No income'] = run_robustness('No income', analysis, formula_extra='')
# Need to adjust — remove hinctnta
try:
    formula_r5 = ('anti_immig_index ~ task_z * C(welfare_regime, Treatment(reference="Nordic"))'
                  ' + agea + age_sq + female + college + urban + C(cntry_wave)')
    m_r5 = smf.ols(formula_r5, data=analysis).fit(
        cov_type='cluster', cov_kwds={'groups': analysis['cntry_wave']})
    lib_key = [k for k in m_r5.params.index if 'Liberal' in k and 'task_z' in k]
    if lib_key:
        k = lib_key[0]
        robustness['No income'] = {
            'coef': m_r5.params[k], 'se': m_r5.bse[k], 'p': m_r5.pvalues[k],
            'ci_lo': m_r5.conf_int().loc[k, 0], 'ci_hi': m_r5.conf_int().loc[k, 1],
            'n': int(m_r5.nobs),
        }
        print(f'  No income: b={m_r5.params[k]:.4f} (p={m_r5.pvalues[k]:.4f})')
except Exception as e:
    print(f'  R5 failed: {e}')

# R6: Without left-right scale
print('\nR6: Without left-right scale')
if 'lrscale' in analysis.columns and analysis['lrscale'].notna().sum() > 1000:
    # Run model WITH lrscale to show it doesn't change results
    analysis_lr = analysis.dropna(subset=['lrscale']).copy()
    robustness['With L-R scale'] = run_robustness('With L-R scale', analysis_lr, extra_controls=' + lrscale')
else:
    print('  lrscale not available or too much missing')

# R7: Jackknife — leave one country out
print('\nR7: Jackknife (leave one country out)')
jack_results = []
for cntry in sorted(analysis['cntry'].unique()):
    data_jack = analysis[analysis['cntry'] != cntry].copy()
    r = run_robustness(f'Excl. {cntry}', data_jack)
    if r is not None:
        jack_results.append({'country_excluded': cntry, **r})

if jack_results:
    jack_df = pd.DataFrame(jack_results)
    robustness['Jackknife min'] = {
        'coef': jack_df['coef'].min(),
        'country': jack_df.loc[jack_df['coef'].idxmin(), 'country_excluded'],
    }
    robustness['Jackknife max'] = {
        'coef': jack_df['coef'].max(),
        'country': jack_df.loc[jack_df['coef'].idxmax(), 'country_excluded'],
    }
    robustness['Jackknife mean'] = {'coef': jack_df['coef'].mean()}
    robustness['Jackknife sd'] = {'coef': jack_df['coef'].std()}
    print(f'\nJackknife range: [{jack_df["coef"].min():.4f}, {jack_df["coef"].max():.4f}]')
    print(f'Jackknife mean: {jack_df["coef"].mean():.4f} (sd={jack_df["coef"].std():.4f})')
    # Save jackknife details
    jack_df.to_csv(TAB_DIR / 'jackknife_details.csv', index=False)

results['robustness'] = robustness

# ============================================================
# SECTION 6: Country-Level CWED vs RTI Slopes (Plot C replacement)
# ============================================================
print('\n' + '='*60)
print('  SECTION 6: Country-Level CWED vs RTI Slopes')
print('='*60)

try:
    # Get country-specific RTI -> anti-immig slopes
    country_slopes = []
    for cntry in sorted(analysis['cntry'].unique()):
        data_c = analysis[analysis['cntry'] == cntry].dropna(subset=['anti_immig_index', 'task_z'])
        if len(data_c) > 50:
            slope, intercept, r, p, se = stats.linregress(data_c['task_z'], data_c['anti_immig_index'])
            regime = data_c['welfare_regime'].mode().iloc[0]
            cwed_val = data_c['cwed_generosity'].iloc[0] if data_c['cwed_generosity'].notna().any() else np.nan
            cwed_ue = data_c['cwed_ue_generosity'].iloc[0] if data_c['cwed_ue_generosity'].notna().any() else np.nan
            cond_val = data_c['conditionality'].iloc[0] if data_c['conditionality'].notna().any() else np.nan
            almp_val = data_c['almp_pmp'].iloc[0] if 'almp_pmp' in data_c.columns and data_c['almp_pmp'].notna().any() else np.nan

            country_slopes.append({
                'cntry': cntry, 'slope': slope, 'se': se, 'p': p, 'n': len(data_c),
                'regime': regime, 'cwed_generosity': cwed_val, 'cwed_ue_generosity': cwed_ue,
                'conditionality': cond_val, 'almp_pmp': almp_val,
            })

    slopes_df = pd.DataFrame(country_slopes)
    slopes_df.to_csv(TAB_DIR / 'country_slopes.csv', index=False)
    print(f'Country slopes computed for {len(slopes_df)} countries')

    # Test CWED generosity vs slopes
    slopes_cwed = slopes_df.dropna(subset=['cwed_generosity'])
    if len(slopes_cwed) >= 5:
        r_gen, p_gen = stats.pearsonr(slopes_cwed['cwed_generosity'], slopes_cwed['slope'])
        print(f'\nCWED generosity vs RTI->anti-immig slope: r={r_gen:.3f}, p={p_gen:.3f}')
        results['cwed_vs_slopes_generosity'] = {'r': r_gen, 'p': p_gen, 'n': len(slopes_cwed)}

    slopes_cwed_ue = slopes_df.dropna(subset=['cwed_ue_generosity'])
    if len(slopes_cwed_ue) >= 5:
        r_ue, p_ue = stats.pearsonr(slopes_cwed_ue['cwed_ue_generosity'], slopes_cwed_ue['slope'])
        print(f'CWED UE generosity vs slope: r={r_ue:.3f}, p={p_ue:.3f}')
        results['cwed_vs_slopes_ue'] = {'r': r_ue, 'p': p_ue, 'n': len(slopes_cwed_ue)}

    slopes_cond = slopes_df.dropna(subset=['conditionality'])
    if len(slopes_cond) >= 5:
        r_cond, p_cond = stats.pearsonr(slopes_cond['conditionality'], slopes_cond['slope'])
        print(f'Conditionality vs slope: r={r_cond:.3f}, p={p_cond:.3f}')
        results['cwed_vs_slopes_conditionality'] = {'r': r_cond, 'p': p_cond, 'n': len(slopes_cond)}

    # Compare with ALMP
    slopes_almp = slopes_df.dropna(subset=['almp_pmp'])
    if len(slopes_almp) >= 5:
        r_almp, p_almp = stats.pearsonr(slopes_almp['almp_pmp'], slopes_almp['slope'])
        print(f'ALMP spending vs slope: r={r_almp:.3f}, p={p_almp:.3f}')
        results['almp_vs_slopes'] = {'r': r_almp, 'p': p_almp, 'n': len(slopes_almp)}

    print(f'\nCountry slopes table:')
    print(slopes_df[['cntry', 'regime', 'slope', 'p', 'n', 'cwed_generosity', 'cwed_ue_generosity']].to_string(index=False))

except Exception as e:
    print(f'Country slopes failed: {e}')
    traceback.print_exc()


# ============================================================
# SECTION 7: Publication-Ready Figures
# ============================================================
print('\n' + '='*60)
print('  SECTION 7: Publication-Ready Figures')
print('='*60)

# --- Figure 2: RTI vs Anti-Immigration by Welfare Regime (THE KEY FIGURE) ---
print('\nFigure 2: RTI vs Anti-Immigration by Regime')
try:
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)

    for idx, regime in enumerate(REGIME_ORDER):
        ax = axes[idx]
        data_r = analysis[analysis['welfare_regime'] == regime].dropna(
            subset=['task_z', 'anti_immig_index']
        )

        # Binned scatter: 15 bins
        bins = pd.qcut(data_r['task_z'], q=15, duplicates='drop')
        binned = data_r.groupby(bins, observed=True).agg(
            x=('task_z', 'mean'),
            y=('anti_immig_index', 'mean'),
            n=('anti_immig_index', 'count'),
        ).reset_index(drop=True)

        # Scatter points
        ax.scatter(binned['x'], binned['y'], color=REGIME_COLORS[regime],
                   s=40, alpha=0.8, edgecolors='white', linewidth=0.5, zorder=3)

        # Linear fit with CI
        slope, intercept, r, p, se = stats.linregress(data_r['task_z'], data_r['anti_immig_index'])
        x_line = np.linspace(data_r['task_z'].min(), data_r['task_z'].max(), 100)
        y_line = intercept + slope * x_line

        # 95% CI for the regression line
        n = len(data_r)
        x_mean = data_r['task_z'].mean()
        ss_x = ((data_r['task_z'] - x_mean) ** 2).sum()
        resid_se = np.sqrt(((data_r['anti_immig_index'] - intercept - slope * data_r['task_z']) ** 2).sum() / (n - 2))
        ci = 1.96 * resid_se * np.sqrt(1/n + (x_line - x_mean)**2 / ss_x)

        ax.plot(x_line, y_line, color=REGIME_COLORS[regime], linewidth=2, zorder=4)
        ax.fill_between(x_line, y_line - ci, y_line + ci,
                        color=REGIME_COLORS[regime], alpha=0.15, zorder=2)

        # Labels
        ax.set_xlabel('RTI (standardized)')
        if idx == 0:
            ax.set_ylabel('Anti-Immigration Index')
        ax.set_title(regime, fontsize=12, fontweight='bold')

        # Annotation: slope and N
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.text(0.05, 0.95, f'b = {slope:.3f}{sig}\nN = {n:,}',
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_sorting_pattern.pdf')
    fig.savefig(FIG_DIR / 'fig2_sorting_pattern.png')
    plt.close()
    print('  Saved fig2_sorting_pattern.pdf/png')
except Exception as e:
    print(f'  Figure 2 failed: {e}')
    traceback.print_exc()


# --- Figure 3: Marginal Effects ---
print('\nFigure 3: Marginal Effects of RTI across Welfare Context')
try:
    # From Model 2: extract marginal effect of 1-SD RTI increase by regime
    if 'model2' in results and 'params' in results['model2']:
        # Base RTI effect (Nordic reference)
        base_key = [k for k in results['model2']['params'] if k == 'task_z'][0]
        base_eff = results['model2']['params'][base_key]['coef']
        base_se = results['model2']['params'][base_key]['se']

        marginal_effects = [{'regime': 'Nordic', 'effect': base_eff, 'se': base_se}]
        for regime in ['Continental', 'Liberal', 'Southern', 'Eastern']:
            int_key = [k for k in results['model2']['params'] if regime in k and 'task_z' in k]
            if int_key:
                int_eff = results['model2']['params'][int_key[0]]['coef']
                int_se = results['model2']['params'][int_key[0]]['se']
                # Total marginal effect = base + interaction
                total = base_eff + int_eff
                # SE of sum (approximate — ignoring covariance)
                total_se = np.sqrt(base_se**2 + int_se**2)
                marginal_effects.append({'regime': regime, 'effect': total, 'se': total_se})

        me_df = pd.DataFrame(marginal_effects)
        me_df['ci_lo'] = me_df['effect'] - 1.96 * me_df['se']
        me_df['ci_hi'] = me_df['effect'] + 1.96 * me_df['se']

        fig, ax = plt.subplots(figsize=(8, 5))
        x_pos = range(len(me_df))
        colors = [REGIME_COLORS[r] for r in me_df['regime']]

        ax.bar(x_pos, me_df['effect'], color=colors, alpha=0.8, width=0.6, zorder=3)
        ax.errorbar(x_pos, me_df['effect'], yerr=1.96*me_df['se'],
                    fmt='none', color='black', capsize=5, linewidth=1.5, zorder=4)
        ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8, zorder=1)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(me_df['regime'], fontsize=11)
        ax.set_ylabel('Marginal Effect of 1-SD RTI Increase\non Anti-Immigration Index')
        ax.set_xlabel('Welfare Regime')

        plt.tight_layout()
        fig.savefig(FIG_DIR / 'fig3_marginal_effects.pdf')
        fig.savefig(FIG_DIR / 'fig3_marginal_effects.png')
        plt.close()
        print('  Saved fig3_marginal_effects.pdf/png')
    else:
        print('  Skipped — Model 2 results not available')
except Exception as e:
    print(f'  Figure 3 failed: {e}')
    traceback.print_exc()


# --- Figure 4: Education Moderation ---
print('\nFigure 4: Education Moderation')
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for panel_idx, (edu_label, edu_val) in enumerate([('Non-College', 1), ('College', 0)]):
        ax = axes[panel_idx]
        data_edu = analysis[analysis['non_college'] == edu_val]

        slopes_edu = []
        for regime in REGIME_ORDER:
            data_re = data_edu[data_edu['welfare_regime'] == regime].dropna(
                subset=['task_z', 'anti_immig_index'])
            if len(data_re) > 50:
                slope, _, _, p, se = stats.linregress(data_re['task_z'], data_re['anti_immig_index'])
                slopes_edu.append({'regime': regime, 'slope': slope, 'se': se, 'p': p, 'n': len(data_re)})

        if slopes_edu:
            sdf = pd.DataFrame(slopes_edu)
            x_pos = range(len(sdf))
            colors = [REGIME_COLORS[r] for r in sdf['regime']]

            ax.bar(x_pos, sdf['slope'], color=colors, alpha=0.8, width=0.6, zorder=3)
            ax.errorbar(x_pos, sdf['slope'], yerr=1.96*sdf['se'],
                        fmt='none', color='black', capsize=5, linewidth=1.5, zorder=4)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(sdf['regime'], fontsize=10, rotation=15)
            ax.set_title(edu_label, fontsize=12, fontweight='bold')
            if panel_idx == 0:
                ax.set_ylabel('RTI -> Anti-Immigration Slope')

            # Annotate slopes
            for i, row in sdf.iterrows():
                ax.text(i, row['slope'] + 1.96*row['se'] + 0.005, f'{row["slope"]:.3f}',
                        ha='center', va='bottom', fontsize=9)

    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.8, zorder=1)
    axes[0].axhline(y=0, color='grey', linestyle='--', linewidth=0.8, zorder=1)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_education_moderator.pdf')
    fig.savefig(FIG_DIR / 'fig4_education_moderator.png')
    plt.close()
    print('  Saved fig4_education_moderator.pdf/png')
except Exception as e:
    print(f'  Figure 4 failed: {e}')
    traceback.print_exc()


# --- Figure 5: Coefficient Stability (Robustness) ---
print('\nFigure 5: Coefficient Stability')
try:
    # Collect robustness results for plotting
    spec_labels = []
    spec_coefs = []
    spec_ci_lo = []
    spec_ci_hi = []

    for label in ['Main', 'Excl. Eastern', 'No income', 'With L-R scale', 'Subj. insecurity']:
        if label in robustness and robustness[label] is not None and 'coef' in robustness[label]:
            spec_labels.append(label)
            spec_coefs.append(robustness[label]['coef'])
            if 'ci_lo' in robustness[label]:
                spec_ci_lo.append(robustness[label]['ci_lo'])
                spec_ci_hi.append(robustness[label]['ci_hi'])
            else:
                spec_ci_lo.append(robustness[label]['coef'] - 1.96 * robustness[label].get('se', 0))
                spec_ci_hi.append(robustness[label]['coef'] + 1.96 * robustness[label].get('se', 0))

    # Add jackknife range
    if 'Jackknife min' in robustness and 'Jackknife max' in robustness:
        spec_labels.append('Jackknife range')
        jack_mean = robustness.get('Jackknife mean', {}).get('coef', 0)
        spec_coefs.append(jack_mean)
        spec_ci_lo.append(robustness['Jackknife min']['coef'])
        spec_ci_hi.append(robustness['Jackknife max']['coef'])

    if spec_labels:
        fig, ax = plt.subplots(figsize=(8, 5))
        y_pos = range(len(spec_labels))

        # Horizontal dot-and-whisker
        for i in range(len(spec_labels)):
            color = '#D6604D' if spec_labels[i] == 'Main' else '#333333'
            ax.plot([spec_ci_lo[i], spec_ci_hi[i]], [i, i], color=color, linewidth=1.5)
            ax.plot(spec_coefs[i], i, 'o', color=color, markersize=8, zorder=5)

        ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(spec_labels, fontsize=11)
        ax.set_xlabel('RTI × Liberal Interaction Coefficient (95% CI)')
        ax.invert_yaxis()

        plt.tight_layout()
        fig.savefig(FIG_DIR / 'fig5_robustness.pdf')
        fig.savefig(FIG_DIR / 'fig5_robustness.png')
        plt.close()
        print('  Saved fig5_robustness.pdf/png')
    else:
        print('  No robustness results to plot')
except Exception as e:
    print(f'  Figure 5 failed: {e}')
    traceback.print_exc()


# --- Figure 6: Country-Level CWED vs RTI Slopes ---
print('\nFigure 6: CWED vs RTI Slopes')
try:
    slopes_plot = slopes_df.dropna(subset=['cwed_generosity']).copy()
    if len(slopes_plot) >= 5:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter with regime colors and country labels
        for regime in REGIME_ORDER:
            data_r = slopes_plot[slopes_plot['regime'] == regime]
            ax.scatter(data_r['cwed_generosity'], data_r['slope'],
                       color=REGIME_COLORS[regime], s=60, alpha=0.8,
                       label=regime, edgecolors='white', linewidth=0.5, zorder=3)
            # Country labels
            for _, row in data_r.iterrows():
                ax.annotate(row['cntry'], (row['cwed_generosity'], row['slope']),
                            xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

        # Fit line
        r_val, p_val = stats.pearsonr(slopes_plot['cwed_generosity'], slopes_plot['slope'])
        slope_fit, int_fit, _, _, _ = stats.linregress(slopes_plot['cwed_generosity'], slopes_plot['slope'])
        x_fit = np.linspace(slopes_plot['cwed_generosity'].min(), slopes_plot['cwed_generosity'].max(), 100)
        ax.plot(x_fit, int_fit + slope_fit * x_fit, 'k--', linewidth=1.5, alpha=0.5, zorder=2)

        ax.set_xlabel('CWED Total Generosity (mean 2005–2011)')
        ax.set_ylabel('Country RTI -> Anti-Immigration Slope')
        ax.legend(frameon=True, framealpha=0.9, fontsize=9)

        # Annotation
        ax.text(0.05, 0.95, f'r = {r_val:.3f}, p = {p_val:.3f}\nN = {len(slopes_plot)} countries',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        plt.tight_layout()
        fig.savefig(FIG_DIR / 'fig6_cwed_vs_slopes.pdf')
        fig.savefig(FIG_DIR / 'fig6_cwed_vs_slopes.png')
        plt.close()
        print('  Saved fig6_cwed_vs_slopes.pdf/png')
    else:
        print('  Insufficient data for Figure 6')
except Exception as e:
    print(f'  Figure 6 failed: {e}')
    traceback.print_exc()


# ============================================================
# SECTION 8: Publication-Ready Tables
# ============================================================
print('\n' + '='*60)
print('  SECTION 8: Publication-Ready Tables')
print('='*60)

# --- Table 1: Summary Statistics ---
print('\nTable 1: Summary Statistics')
try:
    summary_vars = ['anti_immig_index', 'redist_support', 'task', 'agea', 'female',
                    'college', 'hinctnta', 'urban', 'cwed_generosity', 'cwed_ue_generosity']
    if 'radical_right_vote' in analysis.columns:
        summary_vars.append('radical_right_vote')

    # Overall + by regime
    tables = []
    for label, data in [('Overall', analysis)] + [(r, analysis[analysis['welfare_regime'] == r]) for r in REGIME_ORDER]:
        row = {'Group': label, 'N': len(data)}
        for var in summary_vars:
            if var in data.columns:
                row[f'{var}_mean'] = data[var].mean()
                row[f'{var}_sd'] = data[var].std()
                row[f'{var}_n'] = data[var].notna().sum()
        tables.append(row)

    table1 = pd.DataFrame(tables)
    table1.to_csv(TAB_DIR / 'table1_summary_stats.csv', index=False)
    print(f'  Saved table1_summary_stats.csv')
except Exception as e:
    print(f'  Table 1 failed: {e}')

# --- Table 2: Main Results ---
print('\nTable 2: Main Results')
try:
    table2_rows = []

    # Model 1
    if 'model1' in results:
        table2_rows.append({
            'Variable': 'RTI (std)',
            'Model 1': f'{results["model1"]["coef_rti"]:.3f} ({results["model1"]["se_rti"]:.3f})',
        })

    # Models 2, 3, 4 — add interaction terms
    # Save as CSV with coefficients
    table2_data = {}
    for model_name, model_results in [('Model 1', results.get('model1')),
                                        ('Model 2', results.get('model2')),
                                        ('Model 3', results.get('model3')),
                                        ('Model 4', results.get('model4'))]:
        if model_results:
            table2_data[model_name] = model_results

    # Save raw results as JSON-like CSV
    import json
    with open(TAB_DIR / 'table2_main_results.json', 'w') as f:
        json.dump(table2_data, f, indent=2, default=str)
    print(f'  Saved table2_main_results.json')

    # Also create a formatted CSV
    formatted_rows = []
    for model_key, res in results.items():
        if isinstance(res, dict):
            if 'params' in res:
                for param, vals in res['params'].items():
                    if isinstance(vals, dict) and 'coef' in vals:
                        stars = '***' if vals['p'] < 0.001 else '**' if vals['p'] < 0.01 else '*' if vals['p'] < 0.05 else ''
                        formatted_rows.append({
                            'model': model_key,
                            'variable': param,
                            'coefficient': f'{vals["coef"]:.4f}{stars}',
                            'se': f'({vals["se"]:.4f})',
                            'p_value': f'{vals["p"]:.4f}',
                        })
            elif 'coef_rti' in res:
                stars = '***' if res.get('p_rti', 1) < 0.001 else '**' if res.get('p_rti', 1) < 0.01 else '*' if res.get('p_rti', 1) < 0.05 else ''
                formatted_rows.append({
                    'model': model_key,
                    'variable': 'RTI (std)',
                    'coefficient': f'{res["coef_rti"]:.4f}{stars}',
                    'se': f'({res.get("se_rti", 0):.4f})',
                    'p_value': f'{res.get("p_rti", 0):.4f}',
                })

    if formatted_rows:
        pd.DataFrame(formatted_rows).to_csv(TAB_DIR / 'table2_main_results.csv', index=False)
        print(f'  Saved table2_main_results.csv')

except Exception as e:
    print(f'  Table 2 failed: {e}')
    traceback.print_exc()

# --- Table A1: Robustness ---
print('\nTable A1: Robustness')
try:
    rob_rows = []
    for label, res in robustness.items():
        if res is not None and isinstance(res, dict) and 'coef' in res:
            rob_rows.append({
                'specification': label,
                'coef': res['coef'],
                'se': res.get('se', np.nan),
                'p': res.get('p', np.nan),
                'ci_lo': res.get('ci_lo', np.nan),
                'ci_hi': res.get('ci_hi', np.nan),
                'n': res.get('n', np.nan),
            })

    if rob_rows:
        rob_table = pd.DataFrame(rob_rows)
        rob_table.to_csv(TAB_DIR / 'tableA1_robustness.csv', index=False)
        print(f'  Saved tableA1_robustness.csv ({len(rob_rows)} specifications)')
except Exception as e:
    print(f'  Table A1 failed: {e}')

# ============================================================
# SECTION 9: Save Updated Master Dataset
# ============================================================
print('\n' + '='*60)
print('  SECTION 9: Save Updated Master Dataset')
print('='*60)

master.to_csv(ANALYSIS_DIR / 'sorting_mechanism_master_v2.csv', index=False)
print(f'Saved updated master: {master.shape[0]:,} x {master.shape[1]} columns')
print(f'New columns: cwed_generosity, cwed_ue_generosity, cwed_sk_generosity, cwed_ue_coverage, conditionality, radical_right_vote')

# ============================================================
# SECTION 10: Validation Summary
# ============================================================
print('\n' + '='*60)
print('  VALIDATION SUMMARY')
print('='*60)

for step, info in validation_log.items():
    status = info.get('status', 'UNKNOWN')
    print(f'  [{status}] {step}: {info}')

# Save all results
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

with open(ANALYSIS_DIR / 'final_results.json', 'w') as f:
    json.dump(results, f, indent=2, cls=NpEncoder, default=str)
print(f'\nAll results saved to analysis/final_results.json')

print('\n' + '='*60)
print('  PIPELINE COMPLETE')
print('='*60)
