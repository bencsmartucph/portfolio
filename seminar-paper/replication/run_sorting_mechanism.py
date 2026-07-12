"""
Sorting Mechanism Analysis — Standalone Pipeline
=================================================
Implements the full data construction and analysis pipeline for
"The Wrong Politics" seminar paper.

Runs end-to-end: load ESS → merge RTI → construct DVs → merge welfare
→ descriptive analysis → diagnostic models → plots.

Author: Claude (overnight autonomous run, 2026-03-16)
"""

# --- Config ---
import os
import pandas as pd
import numpy as np
import pyreadstat
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for overnight run
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import statsmodels.formula.api as smf
import warnings
import traceback
import sys
import json

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

ROOT = Path(os.environ.get('RESEARCH_MASTER_ROOT', '.'))  # set to your Research_Master checkout; restricted data lives under ROOT/data/raw
ESS_DIR = ROOT / 'data' / 'raw' / 'ESS_csv'
TASK_FILE = ROOT / 'data' / 'raw' / 'shared_isco_task_scores' / 'isco08_3d-task3.csv'
CROSSWALK_FILE = ROOT / 'data' / 'raw' / 'langenkamp_2022' / 'ess_populist_crosswalk.csv'
CPDS_FILE = ROOT / 'data' / 'raw' / 'baccini_2024' / 'Replication V3' / 'Data' / 'Raw Data' / 'CPDS_Aug_2020.dta'
SIWE_FILE = ROOT / 'data' / 'raw' / 'siwe_2017' / 'SIWE_betaMay2017.dta'
GINGRICH_CTX = ROOT / 'data' / 'raw' / 'gingrich_2019' / 'gingrich - RP_Context_Data.dta'
FIG_DIR = ROOT / 'outputs' / 'figures'
TAB_DIR = ROOT / 'outputs' / 'tables'
ANALYSIS_DIR = ROOT / 'analysis'

# Plot style
sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

# Track results for overnight report
report = {
    'sections': {},
    'errors': [],
    'findings': [],
    'data_quality': [],
    'sample': {},
}

def log_section(name, status, detail=''):
    report['sections'][name] = {'status': status, 'detail': detail}
    print(f'\n{"="*60}')
    print(f'  {name}: {status}')
    if detail:
        print(f'  {detail}')
    print(f'{"="*60}')

def log_error(section, error_msg, attempted_fix=''):
    report['errors'].append({
        'section': section, 'error': error_msg, 'fix': attempted_fix
    })
    print(f'  ERROR in {section}: {error_msg}')

def log_finding(finding):
    report['findings'].append(finding)
    print(f'  FINDING: {finding}')

# ============================================================
# SECTION 1: Load ESS Data
# ============================================================
print('\n' + '#'*60)
print('# SECTION 1: Load ESS Data')
print('#'*60)

def load_ess_csv(filepath, usecols=None):
    """Load ESS CSV with multi-encoding fallback per MEMORY.md."""
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            df = pd.read_csv(filepath, encoding=enc, usecols=usecols, low_memory=False)
            return df
        except (UnicodeDecodeError, ValueError):
            continue
    raise ValueError(f'Could not read {filepath} with any encoding')

# Variables to keep
ID_VARS = ['essround', 'cntry', 'idno']
ISCO_VARS = ['isco08', 'iscoco']
DV_EXCLUSION = ['imwbcnt', 'imueclt', 'imbgeco']
DV_SOLIDARITY = ['gincdif']
DV_TRUST = ['trstprl', 'trstplt']
DV_DESERVING = ['sbstrec', 'sbprvpv', 'sbbsntx', 'uentrjb']
VULN_VARS = ['emplno', 'uemp3m', 'uemp12m']
CONTROL_VARS = ['agea', 'gndr', 'eisced', 'eduyrs', 'hinctnta', 'hinctnt',
                'domicil', 'lrscale']

ALL_CANDIDATE_VARS = (ID_VARS + ISCO_VARS + DV_EXCLUSION + DV_SOLIDARITY +
                      DV_TRUST + DV_DESERVING + VULN_VARS + CONTROL_VARS)

try:
    ess_files = sorted(ESS_DIR.glob('*/ESS*.csv'))
    print(f'Found {len(ess_files)} ESS CSV files')

    waves = {}
    wave_summaries = []

    for fpath in ess_files:
        header = pd.read_csv(fpath, nrows=0, encoding='latin-1').columns.tolist()
        available = [v for v in ALL_CANDIDATE_VARS if v in header]
        prtvt_cols = [c for c in header if c.startswith('prtvt')]
        cols_to_load = available + prtvt_cols

        df = load_ess_csv(fpath, usecols=cols_to_load)
        round_num = int(df['essround'].iloc[0])
        waves[round_num] = df

        isco_type = 'isco08' if 'isco08' in df.columns else ('iscoco' if 'iscoco' in df.columns else 'NONE')

        wave_summaries.append({
            'wave': round_num,
            'n_obs': len(df),
            'n_countries': df['cntry'].nunique(),
            'isco_type': isco_type,
            'has_imwbcnt': 'imwbcnt' in df.columns,
            'has_gincdif': 'gincdif' in df.columns,
            'has_sbstrec': 'sbstrec' in df.columns,
            'has_hinctnta': 'hinctnta' in df.columns,
            'n_prtvt_vars': len(prtvt_cols),
        })
        print(f'  Wave {round_num}: {len(df):,} obs, {df["cntry"].nunique()} countries, ISCO={isco_type}')

    # Variable availability matrix
    summary_df = pd.DataFrame(wave_summaries).set_index('wave').sort_index()
    print('\nVariable availability matrix:')
    print(summary_df.to_string())

    # Stack waves 6-9 (have isco08)
    primary_waves = [w for w in [6, 7, 8, 9] if w in waves]
    dfs = [waves[w].copy() for w in primary_waves]
    ess = pd.concat(dfs, ignore_index=True)

    print(f'\nPrimary sample (waves {primary_waves}): {len(ess):,} observations')
    print(f'Countries: {ess["cntry"].nunique()}')
    print(f'Country list: {sorted(ess["cntry"].unique())}')

    report['sample']['total_obs_loaded'] = len(ess)
    report['sample']['waves'] = primary_waves
    report['sample']['n_countries'] = int(ess['cntry'].nunique())
    log_section('1. Load ESS Data', 'SUCCESS',
                f'{len(ess):,} obs from waves {primary_waves}, {ess["cntry"].nunique()} countries')

except Exception as e:
    log_section('1. Load ESS Data', 'FAILED', str(e))
    log_error('Load ESS', traceback.format_exc())
    print('FATAL: Cannot proceed without ESS data.')
    sys.exit(1)

# ============================================================
# SECTION 2: Merge RTI (Automation Exposure)
# ============================================================
print('\n' + '#'*60)
print('# SECTION 2: Merge RTI Scores')
print('#'*60)

try:
    # Load task scores — MEMORY.md: column is `task`, NOT `rtask`/`nrtask`
    tasks = pd.read_csv(TASK_FILE)
    has_task_col = 'task' in tasks.columns
    print(f'Task score file: {tasks.shape[0]} rows, {tasks.shape[1]} columns')
    print(f'Columns: {list(tasks.columns)}')
    print(f'Column "task" present: {has_task_col}')

    if not has_task_col:
        # Try to find the right column
        numeric_cols = tasks.select_dtypes(include=[np.number]).columns.tolist()
        non_id_numeric = [c for c in numeric_cols if c != 'isco08_3d']
        if non_id_numeric:
            task_col_name = non_id_numeric[0]
            print(f'WARNING: "task" column not found. Using "{task_col_name}" instead.')
            tasks = tasks.rename(columns={task_col_name: 'task'})
        else:
            raise ValueError('Cannot identify task score column in task file')

    print(f'Task score range: [{tasks["task"].min():.3f}, {tasks["task"].max():.3f}]')
    print(f'Mean: {tasks["task"].mean():.3f}, SD: {tasks["task"].std():.3f}')

    # Truncate 4-digit ISCO-08 to 3-digit (per MEMORY.md)
    ess['isco08_raw'] = ess['isco08'].copy()
    ess['isco08_valid'] = pd.to_numeric(ess['isco08'], errors='coerce')
    ess.loc[ess['isco08_valid'] < 0, 'isco08_valid'] = np.nan  # ESS negative = missing

    ess['isco08_3d'] = (ess['isco08_valid'] // 10).astype('Int64')

    print(f'\nISCO-08 availability:')
    print(f'  Total observations: {len(ess):,}')
    print(f'  Valid ISCO-08 codes: {ess["isco08_valid"].notna().sum():,} ({ess["isco08_valid"].notna().mean():.1%})')
    print(f'  Unique 3-digit codes in ESS: {ess["isco08_3d"].dropna().nunique()}')
    print(f'  Unique 3-digit codes in task file: {tasks["isco08_3d"].nunique()}')

    # Merge
    n_before = len(ess)
    ess = ess.merge(tasks[['isco08_3d', 'task']], on='isco08_3d', how='left')
    n_after = len(ess)
    assert n_before == n_after, f'ROW INFLATION in RTI merge: {n_before} -> {n_after}'

    has_isco = ess['isco08_valid'].notna()
    has_task = ess['task'].notna()
    overall_match = has_task.mean()
    conditional_match = has_task[has_isco].mean()

    print(f'\nRTI merge results:')
    print(f'  Overall match rate: {overall_match:.1%}')
    print(f'  Match rate among ISCO-coded: {conditional_match:.1%}')
    print(f'  N with RTI scores: {has_task.sum():,}')

    report['sample']['rti_match_overall'] = f'{overall_match:.1%}'
    report['sample']['rti_match_conditional'] = f'{conditional_match:.1%}'
    report['sample']['n_with_rti'] = int(has_task.sum())

    if conditional_match < 0.50:
        log_error('RTI merge', f'Match rate only {conditional_match:.1%} — suspiciously low')
    else:
        print(f'  Match rate looks good ({conditional_match:.1%})')

    # RTI distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ess['task'].dropna().hist(bins=30, ax=axes[0], edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('RTI Score (task)')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(ess['task'].mean(), color='red', linestyle='--',
                    label=f'Mean={ess["task"].mean():.2f}')
    axes[0].legend()

    top_countries = ess.groupby('cntry')['task'].count().nlargest(10).index
    ess[ess['cntry'].isin(top_countries)].boxplot(column='task', by='cntry', ax=axes[1])
    axes[1].set_xlabel('Country')
    axes[1].set_ylabel('RTI Score')
    plt.suptitle('')
    axes[1].set_title('')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'rti_distribution.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'rti_distribution.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved: outputs/figures/rti_distribution.pdf + .png')

    log_section('2. Merge RTI', 'SUCCESS',
                f'Match rate {conditional_match:.1%}, N={has_task.sum():,}')

except Exception as e:
    log_section('2. Merge RTI', 'FAILED', str(e))
    log_error('RTI merge', traceback.format_exc())

# ============================================================
# SECTION 3: Construct Dependent Variables
# ============================================================
print('\n' + '#'*60)
print('# SECTION 3: Construct DVs')
print('#'*60)

# --- 3a. Anti-immigration index ---
try:
    immig_vars = ['imwbcnt', 'imueclt', 'imbgeco']
    for v in immig_vars:
        if v in ess.columns:
            ess[v] = pd.to_numeric(ess[v], errors='coerce')
            ess.loc[ess[v] > 10, v] = np.nan
            ess.loc[ess[v] < 0, v] = np.nan

    for v in immig_vars:
        if v in ess.columns:
            valid = ess[v].notna().sum()
            print(f'{v}: {valid:,} valid ({ess[v].notna().mean():.1%}), mean={ess[v].mean():.2f}')

    # Reverse-code: higher = more anti-immigration
    for v in immig_vars:
        if v in ess.columns:
            ess[f'{v}_rev'] = 10 - ess[v]

    rev_vars = [f'{v}_rev' for v in immig_vars if v in ess.columns]
    ess['anti_immig_index'] = ess[rev_vars].mean(axis=1, skipna=True)
    ess.loc[ess[rev_vars].notna().sum(axis=1) < 2, 'anti_immig_index'] = np.nan

    print(f'\nAnti-immigration index: {ess["anti_immig_index"].notna().sum():,} valid')
    print(f'  Mean={ess["anti_immig_index"].mean():.2f}, SD={ess["anti_immig_index"].std():.2f}')
    print(f'  Range: [{ess["anti_immig_index"].min():.1f}, {ess["anti_immig_index"].max():.1f}]')

    # Cronbach's alpha
    def cronbach_alpha(df_items):
        df_clean = df_items.dropna()
        k = df_clean.shape[1]
        if k < 2 or len(df_clean) < 10:
            return np.nan
        item_vars = df_clean.var(axis=0, ddof=1)
        total_var = df_clean.sum(axis=1).var(ddof=1)
        return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

    alpha = cronbach_alpha(ess[rev_vars])
    print(f'  Cronbach alpha: {alpha:.3f}')
    log_finding(f'Anti-immigration index alpha = {alpha:.3f}')

except Exception as e:
    log_error('3a Anti-immigration', traceback.format_exc())

# --- 3b. Redistribution support ---
try:
    ess['gincdif'] = pd.to_numeric(ess['gincdif'], errors='coerce')
    ess.loc[(ess['gincdif'] > 5) | (ess['gincdif'] < 1), 'gincdif'] = np.nan
    # Reverse so higher = more support for redistribution
    ess['redist_support'] = 6 - ess['gincdif']

    print(f'\nRedistribution support: {ess["redist_support"].notna().sum():,} valid')
    print(f'  Mean={ess["redist_support"].mean():.2f}, SD={ess["redist_support"].std():.2f}')

except Exception as e:
    log_error('3b Redistribution', traceback.format_exc())

# --- 3c. Deservingness items (wave 8 only) ---
try:
    deserv_vars = ['sbstrec', 'sbprvpv', 'sbbsntx', 'uentrjb']
    deserv_present = [v for v in deserv_vars if v in ess.columns]
    print(f'\nDeservingness items present: {deserv_present}')

    for v in deserv_present:
        ess[v] = pd.to_numeric(ess[v], errors='coerce')
        ess.loc[(ess[v] > 5) | (ess[v] < 1), v] = np.nan
        valid_in_w8 = ess.loc[ess['essround'] == 8, v].notna().sum()
        print(f'  {v}: {valid_in_w8:,} valid in wave 8')

    if len(deserv_present) >= 2:
        # Narrow deservingness: higher = more INCLUSIVE (raw ESS items, 1=agree..5=disagree;
        # low end = agreeing benefits strain the economy / unemployed don't try = restrictive view).
        # [direction corrected 2026-06-14, anchored to redist_support; see two_faces_of_solidarity/CORRECTION_NOTE.md]
        narrow_items = [v for v in ['sbstrec', 'uentrjb'] if v in deserv_present]
        if narrow_items:
            ess['narrow_deserving'] = ess[narrow_items].mean(axis=1, skipna=True)
            ess.loc[ess[narrow_items].notna().sum(axis=1) < 1, 'narrow_deserving'] = np.nan
            w8_valid = ess.loc[ess['essround'] == 8, 'narrow_deserving'].notna().sum()
            print(f'  narrow_deserving index: {w8_valid:,} valid in wave 8')

except Exception as e:
    log_error('3c Deservingness', traceback.format_exc())

# --- 3d. Trust variables ---
try:
    for v in ['trstprl', 'trstplt']:
        if v in ess.columns:
            ess[v] = pd.to_numeric(ess[v], errors='coerce')
            ess.loc[(ess[v] > 10) | (ess[v] < 0), v] = np.nan
            print(f'{v}: {ess[v].notna().sum():,} valid, mean={ess[v].mean():.2f}')

except Exception as e:
    log_error('3d Trust', traceback.format_exc())

log_section('3. Construct DVs', 'SUCCESS')

# ============================================================
# SECTION 4: Clean Controls
# ============================================================
print('\n' + '#'*60)
print('# SECTION 4: Clean Controls')
print('#'*60)

try:
    ess['agea'] = pd.to_numeric(ess['agea'], errors='coerce')
    ess.loc[(ess['agea'] > 110) | (ess['agea'] < 15), 'agea'] = np.nan
    ess['age_sq'] = ess['agea'] ** 2

    ess['gndr'] = pd.to_numeric(ess['gndr'], errors='coerce')
    ess['female'] = (ess['gndr'] == 2).astype(float)
    ess.loc[ess['gndr'].isna() | (ess['gndr'] > 2), 'female'] = np.nan

    ess['eisced'] = pd.to_numeric(ess['eisced'], errors='coerce')
    ess.loc[(ess['eisced'] > 7) | (ess['eisced'] < 1), 'eisced'] = np.nan

    ess['eduyrs'] = pd.to_numeric(ess['eduyrs'], errors='coerce')
    ess.loc[(ess['eduyrs'] > 40) | (ess['eduyrs'] < 0), 'eduyrs'] = np.nan

    ess['hinctnta'] = pd.to_numeric(ess['hinctnta'], errors='coerce')
    ess.loc[(ess['hinctnta'] > 10) | (ess['hinctnta'] < 1), 'hinctnta'] = np.nan

    ess['domicil'] = pd.to_numeric(ess['domicil'], errors='coerce')
    ess.loc[(ess['domicil'] > 5) | (ess['domicil'] < 1), 'domicil'] = np.nan
    ess['urban'] = (ess['domicil'] <= 2).astype(float)
    ess.loc[ess['domicil'].isna(), 'urban'] = np.nan

    ess['lrscale'] = pd.to_numeric(ess['lrscale'], errors='coerce')
    ess.loc[(ess['lrscale'] > 10) | (ess['lrscale'] < 0), 'lrscale'] = np.nan

    # Education binary for moderator analysis
    ess['college'] = (ess['eisced'] >= 6).astype(float)
    ess.loc[ess['eisced'].isna(), 'college'] = np.nan

    print('Controls cleaned:')
    for v in ['agea', 'female', 'eisced', 'eduyrs', 'hinctnta', 'urban', 'lrscale', 'college']:
        if v in ess.columns:
            print(f'  {v}: {ess[v].notna().sum():,} valid ({ess[v].notna().mean():.1%})')

    log_section('4. Clean Controls', 'SUCCESS')

except Exception as e:
    log_section('4. Clean Controls', 'FAILED', str(e))
    log_error('Controls', traceback.format_exc())

# ============================================================
# SECTION 5: Merge Welfare Indicators (CPDS)
# ============================================================
print('\n' + '#'*60)
print('# SECTION 5: Welfare Indicators')
print('#'*60)

cpds_merged = False
try:
    cpds, cpds_meta = pyreadstat.read_dta(str(CPDS_FILE))
    print(f'CPDS: {cpds.shape[0]} rows x {cpds.shape[1]} columns')
    print(f'Years: {cpds["year"].min():.0f}-{cpds["year"].max():.0f}')

    # Find welfare-relevant columns
    welfare_keywords = ['almp', 'unemp', 'socexp', 'emprot', 'train', 'incent', 'jobcrea']
    welfare_cols = [c for c in cpds.columns
                   if any(kw in c.lower() for kw in welfare_keywords)]
    print(f'Welfare-relevant columns: {len(welfare_cols)}')
    for col in sorted(welfare_cols)[:15]:
        label = cpds_meta.column_names_to_labels.get(col, '')
        print(f'  {col}: {cpds[col].notna().sum()} valid — {label[:60]}')

    # Select key indicators
    cpds_key = ['country', 'year']
    for c in ['almp_pmp', 'unemp_pmp', 'socexp_t_pmp', 'emprot_reg', 'emprot_temp',
              'training_pmp', 'incent_pmp', 'jobcrea_pmp']:
        if c in cpds.columns:
            cpds_key.append(c)

    cpds_welfare = cpds[cpds_key].copy()
    cpds_ess = cpds_welfare[(cpds_welfare['year'] >= 2002) & (cpds_welfare['year'] <= 2019)].copy()

    # Construct ratios
    if 'almp_pmp' in cpds_ess.columns and 'unemp_pmp' in cpds_ess.columns:
        denom = cpds_ess['almp_pmp'] + cpds_ess['unemp_pmp']
        cpds_ess['active_passive_ratio'] = cpds_ess['almp_pmp'] / denom.replace(0, np.nan)

    if 'training_pmp' in cpds_ess.columns and 'almp_pmp' in cpds_ess.columns:
        cpds_ess['training_share'] = cpds_ess['training_pmp'] / cpds_ess['almp_pmp'].replace(0, np.nan)

    if 'incent_pmp' in cpds_ess.columns and 'almp_pmp' in cpds_ess.columns:
        cpds_ess['incentive_share'] = cpds_ess['incent_pmp'] / cpds_ess['almp_pmp'].replace(0, np.nan)

    print(f'\nCPDS for ESS period: {len(cpds_ess)} rows, {cpds_ess["country"].nunique()} countries')

    # Map ESS round -> fieldwork year
    round_to_year = {1: 2002, 2: 2004, 3: 2006, 4: 2008, 5: 2010,
                     6: 2012, 7: 2014, 8: 2016, 9: 2018}
    ess['fieldwork_year'] = ess['essround'].map(round_to_year)

    # Map CPDS country names to ESS ISO-2 codes
    cpds_to_ess = {
        'Australia': 'AU', 'Austria': 'AT', 'Belgium': 'BE', 'Bulgaria': 'BG',
        'Canada': 'CA', 'Croatia': 'HR', 'Cyprus': 'CY', 'Czech Republic': 'CZ',
        'Denmark': 'DK', 'Estonia': 'EE', 'Finland': 'FI', 'France': 'FR',
        'Germany': 'DE', 'Greece': 'GR', 'Hungary': 'HU', 'Iceland': 'IS',
        'Ireland': 'IE', 'Israel': 'IL', 'Italy': 'IT', 'Japan': 'JP',
        'Latvia': 'LV', 'Lithuania': 'LT', 'Luxembourg': 'LU',
        'Netherlands': 'NL', 'New Zealand': 'NZ', 'Norway': 'NO',
        'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO',
        'Slovakia': 'SK', 'Slovenia': 'SI', 'Spain': 'ES',
        'Sweden': 'SE', 'Switzerland': 'CH', 'Turkey': 'TR',
        'United Kingdom': 'GB', 'United States': 'US'
    }
    cpds_ess['cntry'] = cpds_ess['country'].map(cpds_to_ess)

    welfare_merge_cols = ['cntry', 'year']
    for c in ['almp_pmp', 'unemp_pmp', 'socexp_t_pmp', 'active_passive_ratio',
              'training_share', 'incentive_share']:
        if c in cpds_ess.columns:
            welfare_merge_cols.append(c)

    welfare_for_merge = cpds_ess[welfare_merge_cols].dropna(subset=['cntry']).copy()

    n_before = len(ess)
    ess = ess.merge(
        welfare_for_merge.rename(columns={'year': 'fieldwork_year'}),
        on=['cntry', 'fieldwork_year'],
        how='left'
    )
    n_after = len(ess)
    assert n_before == n_after, f'Row inflation in welfare merge: {n_before} -> {n_after}'

    welfare_match = ess['almp_pmp'].notna().mean() if 'almp_pmp' in ess.columns else 0
    print(f'\nWelfare indicator match rate: {welfare_match:.1%}')
    print(f'Countries with ALMP data: {ess.loc[ess["almp_pmp"].notna(), "cntry"].nunique()}')
    cpds_merged = True

    log_section('5. Welfare Indicators', 'SUCCESS', f'CPDS match rate {welfare_match:.1%}')

except Exception as e:
    log_section('5. Welfare Indicators', 'PARTIAL',
                f'CPDS merge failed: {str(e)[:100]}. Will use regime classification only.')
    log_error('CPDS merge', traceback.format_exc())

# ============================================================
# SECTION 5b: Welfare Regime Classification
# ============================================================
welfare_regimes = {
    'Nordic': ['DK', 'SE', 'NO', 'FI', 'IS'],
    'Continental': ['DE', 'FR', 'AT', 'BE', 'NL', 'CH', 'LU'],
    'Liberal': ['GB', 'IE'],
    'Southern': ['ES', 'IT', 'PT', 'GR', 'CY'],
    'Eastern': ['CZ', 'PL', 'HU', 'SK', 'SI', 'EE', 'LT', 'LV',
                'BG', 'HR', 'RO', 'RS', 'UA', 'AL', 'ME', 'XK'],
}

regime_map = {}
for regime, countries in welfare_regimes.items():
    for c in countries:
        regime_map[c] = regime

ess['welfare_regime'] = ess['cntry'].map(regime_map)
unclassified = ess.loc[ess['welfare_regime'].isna(), 'cntry'].unique()
if len(unclassified) > 0:
    print(f'Countries without regime classification: {sorted(unclassified)}')
    ess.loc[ess['welfare_regime'].isna(), 'welfare_regime'] = 'Other'

print('\nObservations by welfare regime:')
print(ess['welfare_regime'].value_counts().sort_index())

ess['cntry_wave'] = ess['cntry'] + '_' + ess['essround'].astype(str)

# ============================================================
# SECTION 6: THE KEY PLOTS
# ============================================================
print('\n' + '#'*60)
print('# SECTION 6: Key Diagnostic Plots')
print('#'*60)

# --- Plot A: RTI vs anti-immigration by regime (THE CRITICAL PLOT) ---
try:
    plot_regimes = ['Nordic', 'Continental', 'Liberal', 'Southern', 'Eastern']
    plot_df = ess[
        ess['welfare_regime'].isin(plot_regimes) &
        ess['task'].notna() &
        ess['anti_immig_index'].notna()
    ].copy()

    print(f'Plot A observations: {len(plot_df):,}')
    for r in plot_regimes:
        n = (plot_df['welfare_regime'] == r).sum()
        print(f'  {r}: {n:,}')

    # Determine grid layout based on number of regimes with data
    regimes_with_data = [r for r in plot_regimes
                         if (plot_df['welfare_regime'] == r).sum() >= 100]
    n_regimes = len(regimes_with_data)

    if n_regimes <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
        axes_flat = axes.flat
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
        axes_flat = axes.flat

    regime_colors = {'Nordic': '#2196F3', 'Continental': '#FF9800',
                     'Liberal': '#F44336', 'Southern': '#4CAF50',
                     'Eastern': '#9C27B0'}

    slopes_for_report = {}

    for idx, regime in enumerate(regimes_with_data):
        if idx >= len(list(axes_flat)):
            break
        ax = list(axes_flat)[idx]
        rdf = plot_df[plot_df['welfare_regime'] == regime].copy()

        # Binned scatter (ventiles)
        n_bins = min(20, max(5, len(rdf) // 100))
        rdf['rti_bin'] = pd.qcut(rdf['task'], q=n_bins, duplicates='drop')
        binned = rdf.groupby('rti_bin', observed=True).agg(
            rti_mean=('task', 'mean'),
            dv_mean=('anti_immig_index', 'mean'),
            dv_se=('anti_immig_index', lambda x: x.std() / np.sqrt(len(x))),
            n=('anti_immig_index', 'count')
        ).reset_index()

        color = regime_colors.get(regime, 'gray')
        ax.errorbar(binned['rti_mean'], binned['dv_mean'],
                    yerr=1.96 * binned['dv_se'], fmt='o', markersize=5,
                    capsize=2, alpha=0.8, color=color)

        # OLS line
        valid = rdf[['task', 'anti_immig_index']].dropna()
        slope, intercept, r_val, p_val, se = stats.linregress(
            valid['task'], valid['anti_immig_index'])
        x_range = np.linspace(rdf['task'].min(), rdf['task'].max(), 100)
        ax.plot(x_range, intercept + slope * x_range, '--', color=color, alpha=0.7,
                label=f'slope={slope:.3f} (p={p_val:.3f})')

        slopes_for_report[regime] = {'slope': slope, 'se': se, 'p': p_val,
                                      'r': r_val, 'n': len(valid)}

        ax.set_xlabel('RTI Score (automation exposure)')
        ax.set_ylabel('Anti-immigration index (0-10)')
        ax.set_title(f'{regime} (N={len(rdf):,})')
        ax.legend(fontsize=9)

    # Hide unused axes
    for idx in range(len(regimes_with_data), len(list(axes_flat))):
        list(axes_flat)[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig2_rti_vs_antiimmig_by_regime.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'fig2_rti_vs_antiimmig_by_regime.png', bbox_inches='tight', dpi=150)
    plt.close()

    print('\nPlot A slopes by regime:')
    for regime, vals in slopes_for_report.items():
        direction = 'POSITIVE' if vals['slope'] > 0 else 'NEGATIVE'
        sig = 'significant' if vals['p'] < 0.05 else 'NOT significant'
        print(f'  {regime}: slope={vals["slope"]:.4f} (SE={vals["se"]:.4f}, p={vals["p"]:.4f}) — {direction}, {sig}')
        log_finding(f'Plot A {regime}: slope={vals["slope"]:.4f}, p={vals["p"]:.4f} ({direction}, {sig})')

    print('\nSaved: outputs/figures/fig2_rti_vs_antiimmig_by_regime.pdf + .png')
    log_section('6a. Plot A (RTI vs anti-immig)', 'SUCCESS')

except Exception as e:
    log_section('6a. Plot A', 'FAILED', str(e))
    log_error('Plot A', traceback.format_exc())

# --- Plot B: RTI vs redistribution by regime ---
try:
    plot_df2 = ess[
        ess['welfare_regime'].isin(plot_regimes) &
        ess['task'].notna() &
        ess['redist_support'].notna()
    ].copy()

    print(f'\nPlot B observations: {len(plot_df2):,}')

    regimes_with_data_b = [r for r in plot_regimes
                           if (plot_df2['welfare_regime'] == r).sum() >= 100]
    n_reg_b = len(regimes_with_data_b)

    if n_reg_b <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
        axes_flat = axes.flat
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
        axes_flat = axes.flat

    slopes_b = {}
    for idx, regime in enumerate(regimes_with_data_b):
        if idx >= len(list(axes_flat)):
            break
        ax = list(axes_flat)[idx]
        rdf = plot_df2[plot_df2['welfare_regime'] == regime].copy()

        n_bins = min(20, max(5, len(rdf) // 100))
        rdf['rti_bin'] = pd.qcut(rdf['task'], q=n_bins, duplicates='drop')
        binned = rdf.groupby('rti_bin', observed=True).agg(
            rti_mean=('task', 'mean'),
            dv_mean=('redist_support', 'mean'),
            dv_se=('redist_support', lambda x: x.std() / np.sqrt(len(x))),
        ).reset_index()

        color = regime_colors.get(regime, 'gray')
        ax.errorbar(binned['rti_mean'], binned['dv_mean'],
                    yerr=1.96 * binned['dv_se'], fmt='o', markersize=5,
                    capsize=2, alpha=0.8, color=color)

        valid = rdf[['task', 'redist_support']].dropna()
        slope, intercept, r_val, p_val, se = stats.linregress(
            valid['task'], valid['redist_support'])
        x_range = np.linspace(rdf['task'].min(), rdf['task'].max(), 100)
        ax.plot(x_range, intercept + slope * x_range, '--', color=color, alpha=0.7,
                label=f'slope={slope:.3f} (p={p_val:.3f})')

        slopes_b[regime] = {'slope': slope, 'se': se, 'p': p_val, 'n': len(valid)}

        ax.set_xlabel('RTI Score (automation exposure)')
        ax.set_ylabel('Redistribution support (1-5)')
        ax.set_title(f'{regime} (N={len(rdf):,})')
        ax.legend(fontsize=9)

    for idx in range(len(regimes_with_data_b), len(list(axes_flat))):
        list(axes_flat)[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig4_rti_vs_redistribution_by_regime.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'fig4_rti_vs_redistribution_by_regime.png', bbox_inches='tight', dpi=150)
    plt.close()

    print('\nPlot B slopes by regime:')
    for regime, vals in slopes_b.items():
        direction = 'POSITIVE' if vals['slope'] > 0 else 'NEGATIVE'
        sig = 'significant' if vals['p'] < 0.05 else 'NOT significant'
        print(f'  {regime}: slope={vals["slope"]:.4f} (p={vals["p"]:.4f}) — {direction}, {sig}')
        log_finding(f'Plot B {regime}: slope={vals["slope"]:.4f}, p={vals["p"]:.4f} ({direction}, {sig})')

    print('\nSaved: outputs/figures/fig4_rti_vs_redistribution_by_regime.pdf + .png')
    log_section('6b. Plot B (RTI vs redistribution)', 'SUCCESS')

except Exception as e:
    log_section('6b. Plot B', 'FAILED', str(e))
    log_error('Plot B', traceback.format_exc())

# --- Plot C: Country-level welfare vs RTI slopes ---
try:
    slopes_list = []
    for (cntry, wave), group in ess.groupby(['cntry', 'essround']):
        valid = group[['task', 'anti_immig_index']].dropna()
        if len(valid) >= 50:
            slope, intercept, r_val, p_val, se = stats.linregress(
                valid['task'], valid['anti_immig_index'])
            row = {
                'cntry': cntry, 'essround': wave,
                'rti_slope': slope, 'rti_slope_se': se, 'rti_p': p_val,
                'n': len(valid),
                'regime': group['welfare_regime'].iloc[0],
            }
            # Add welfare indicators if available
            for wv in ['almp_pmp', 'active_passive_ratio', 'socexp_t_pmp']:
                if wv in group.columns:
                    row[wv] = group[wv].iloc[0]
            slopes_list.append(row)

    slopes_df = pd.DataFrame(slopes_list)
    print(f'\nCountry-wave slope estimates: {len(slopes_df)}')

    # Try plotting against ALMP if available
    has_almp = 'almp_pmp' in slopes_df.columns and slopes_df['almp_pmp'].notna().sum() > 5
    has_apr = 'active_passive_ratio' in slopes_df.columns and slopes_df['active_passive_ratio'].notna().sum() > 5

    if has_almp or has_apr:
        n_panels = sum([has_almp, has_apr])
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))
        if n_panels == 1:
            axes = [axes]

        panel_idx = 0

        if has_almp:
            ax = axes[panel_idx]
            plot_s = slopes_df.dropna(subset=['almp_pmp'])
            for regime in plot_s['regime'].unique():
                rdf = plot_s[plot_s['regime'] == regime]
                color = regime_colors.get(regime, 'gray')
                ax.scatter(rdf['almp_pmp'], rdf['rti_slope'],
                          c=color, label=regime, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
                for _, row in rdf.iterrows():
                    ax.annotate(row['cntry'], (row['almp_pmp'], row['rti_slope']),
                               fontsize=7, alpha=0.7)

            # Overall correlation line
            valid_s = plot_s[['almp_pmp', 'rti_slope']].dropna()
            if len(valid_s) > 3:
                sl, interc, r, p, se = stats.linregress(valid_s['almp_pmp'], valid_s['rti_slope'])
                x_r = np.linspace(valid_s['almp_pmp'].min(), valid_s['almp_pmp'].max(), 100)
                ax.plot(x_r, interc + sl * x_r, 'k--', alpha=0.5,
                        label=f'r={r:.2f}, p={p:.3f}')
                log_finding(f'Plot C ALMP correlation: r={r:.2f}, p={p:.3f}')

            ax.set_xlabel('ALMP spending (% GDP)')
            ax.set_ylabel('RTI -> anti-immigration slope')
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax.legend(fontsize=8)
            panel_idx += 1

        if has_apr:
            ax = axes[panel_idx]
            plot_s = slopes_df.dropna(subset=['active_passive_ratio'])
            for regime in plot_s['regime'].unique():
                rdf = plot_s[plot_s['regime'] == regime]
                color = regime_colors.get(regime, 'gray')
                ax.scatter(rdf['active_passive_ratio'], rdf['rti_slope'],
                          c=color, label=regime, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
                for _, row in rdf.iterrows():
                    ax.annotate(row['cntry'], (row['active_passive_ratio'], row['rti_slope']),
                               fontsize=7, alpha=0.7)

            valid_s = plot_s[['active_passive_ratio', 'rti_slope']].dropna()
            if len(valid_s) > 3:
                sl, interc, r, p, se = stats.linregress(valid_s['active_passive_ratio'], valid_s['rti_slope'])
                x_r = np.linspace(valid_s['active_passive_ratio'].min(),
                                  valid_s['active_passive_ratio'].max(), 100)
                ax.plot(x_r, interc + sl * x_r, 'k--', alpha=0.5,
                        label=f'r={r:.2f}, p={p:.3f}')
                log_finding(f'Plot C active/passive ratio correlation: r={r:.2f}, p={p:.3f}')

            ax.set_xlabel('Active/Passive spending ratio')
            ax.set_ylabel('RTI -> anti-immigration slope')
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(FIG_DIR / 'welfare_vs_rti_slopes.pdf', bbox_inches='tight')
        plt.savefig(FIG_DIR / 'welfare_vs_rti_slopes.png', bbox_inches='tight', dpi=150)
        plt.close()
        print('Saved: outputs/figures/welfare_vs_rti_slopes.pdf + .png')
        log_section('6c. Plot C (welfare vs slopes)', 'SUCCESS')
    else:
        # Fallback: plot slopes by regime type
        fig, ax = plt.subplots(figsize=(10, 6))
        regime_order = ['Nordic', 'Continental', 'Liberal', 'Southern', 'Eastern']
        regime_order = [r for r in regime_order if r in slopes_df['regime'].values]

        for regime in regime_order:
            rdf = slopes_df[slopes_df['regime'] == regime]
            color = regime_colors.get(regime, 'gray')
            ax.scatter([regime] * len(rdf), rdf['rti_slope'],
                      c=color, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
            for _, row in rdf.iterrows():
                ax.annotate(row['cntry'], (regime, row['rti_slope']),
                           fontsize=7, alpha=0.7)

        ax.set_ylabel('RTI -> anti-immigration slope')
        ax.set_xlabel('Welfare regime')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'welfare_vs_rti_slopes.pdf', bbox_inches='tight')
        plt.savefig(FIG_DIR / 'welfare_vs_rti_slopes.png', bbox_inches='tight', dpi=150)
        plt.close()
        print('Saved: outputs/figures/welfare_vs_rti_slopes.pdf + .png (regime-based fallback)')
        log_section('6c. Plot C (welfare vs slopes)', 'PARTIAL — regime-based fallback, no continuous welfare data')

except Exception as e:
    log_section('6c. Plot C', 'FAILED', str(e))
    log_error('Plot C', traceback.format_exc())

# ============================================================
# SECTION 7: Summary Statistics
# ============================================================
print('\n' + '#'*60)
print('# SECTION 7: Summary Statistics')
print('#'*60)

try:
    analysis_vars = ['task', 'anti_immig_index', 'redist_support', 'agea',
                     'female', 'eduyrs', 'hinctnta', 'lrscale']
    analysis_vars = [v for v in analysis_vars if v in ess.columns]

    # Overall summary
    overall_stats = ess[analysis_vars].describe().T
    overall_stats['N_valid'] = ess[analysis_vars].notna().sum()
    overall_stats['pct_missing'] = (1 - ess[analysis_vars].notna().mean()) * 100

    print('\nOverall summary statistics:')
    print(overall_stats[['N_valid', 'mean', 'std', 'min', 'max', 'pct_missing']].round(2).to_string())

    # By regime
    regime_stats = []
    for regime in ['Nordic', 'Continental', 'Liberal', 'Southern', 'Eastern']:
        rdf = ess[ess['welfare_regime'] == regime]
        if len(rdf) == 0:
            continue
        row = {'regime': regime, 'N': len(rdf)}
        for v in analysis_vars:
            row[f'{v}_mean'] = rdf[v].mean()
            row[f'{v}_sd'] = rdf[v].std()
            row[f'{v}_N'] = rdf[v].notna().sum()
        regime_stats.append(row)

    regime_stats_df = pd.DataFrame(regime_stats)

    # Save summary stats CSV
    summary_csv = pd.DataFrame()
    for v in analysis_vars:
        for regime in ['Nordic', 'Continental', 'Liberal', 'Southern', 'Eastern']:
            rdf = ess[ess['welfare_regime'] == regime]
            if len(rdf) == 0:
                continue
            summary_csv = pd.concat([summary_csv, pd.DataFrame([{
                'variable': v,
                'regime': regime,
                'mean': rdf[v].mean(),
                'sd': rdf[v].std(),
                'min': rdf[v].min(),
                'max': rdf[v].max(),
                'N': int(rdf[v].notna().sum()),
            }])], ignore_index=True)

    summary_csv.to_csv(TAB_DIR / 'summary_stats.csv', index=False)
    print('\nSaved: outputs/tables/summary_stats.csv')
    log_section('7. Summary Statistics', 'SUCCESS')

except Exception as e:
    log_section('7. Summary Statistics', 'FAILED', str(e))
    log_error('Summary stats', traceback.format_exc())

# ============================================================
# SECTION 8: Diagnostic OLS Models
# ============================================================
print('\n' + '#'*60)
print('# SECTION 8: Diagnostic OLS Models')
print('#'*60)

try:
    # Prepare analysis sample
    h1_vars = ['task', 'anti_immig_index', 'agea', 'age_sq', 'female',
               'eisced', 'hinctnta', 'urban', 'cntry_wave', 'cntry', 'essround',
               'welfare_regime']
    h1_vars_present = [v for v in h1_vars if v in ess.columns]
    h1_sample = ess[h1_vars_present].dropna(subset=[v for v in h1_vars_present
                                                     if v not in ['cntry_wave', 'cntry', 'essround', 'welfare_regime']]).copy()

    print(f'H1 analysis sample: {len(h1_sample):,}')
    print(f'  Countries: {h1_sample["cntry"].nunique()}')
    print(f'  Country-waves: {h1_sample["cntry_wave"].nunique()}')

    report['sample']['h1_sample'] = len(h1_sample)

    # Model 1b: RTI x welfare regime (categorical)
    h1b_sample = h1_sample[h1_sample['welfare_regime'].isin(
        ['Nordic', 'Continental', 'Liberal', 'Southern'])].copy()

    if len(h1b_sample) > 1000:
        for v in ['task', 'agea', 'hinctnta']:
            h1b_sample[f'{v}_z'] = (h1b_sample[v] - h1b_sample[v].mean()) / h1b_sample[v].std()
        h1b_sample['age_sq_z'] = h1b_sample['agea_z'] ** 2

        h1b_sample['welfare_regime'] = pd.Categorical(
            h1b_sample['welfare_regime'],
            categories=['Nordic', 'Continental', 'Liberal', 'Southern']
        )

        print('\nFitting Model 1b: Anti-immigration ~ RTI x Welfare regime (mixed model)...')
        try:
            m1b = smf.mixedlm(
                'anti_immig_index ~ task_z * C(welfare_regime, Treatment(reference="Nordic")) + agea_z + age_sq_z + female + eisced + hinctnta_z + urban',
                data=h1b_sample,
                groups=h1b_sample['cntry_wave']
            ).fit(reml=True)

            print('\nModel 1b Results:')
            print(m1b.summary())

            interaction_terms = [k for k in m1b.params.index if 'task_z:' in k]
            print('\nKey interactions (vs. Nordic):')
            for term in interaction_terms:
                coef = m1b.params[term]
                pval = m1b.pvalues[term]
                sig = '*' if pval < 0.05 else ''
                print(f'  {term}: {coef:.4f} (p={pval:.4f}) {sig}')
                log_finding(f'Model 1b {term}: coef={coef:.4f}, p={pval:.4f}')

            # RTI main effect (Nordic baseline)
            rti_main = m1b.params.get('task_z', None)
            if rti_main is not None:
                print(f'\nRTI main effect (Nordic): {rti_main:.4f} (p={m1b.pvalues["task_z"]:.4f})')
                log_finding(f'Model 1b RTI main (Nordic baseline): {rti_main:.4f}, p={m1b.pvalues["task_z"]:.4f}')

        except Exception as e:
            print(f'Mixed model failed: {e}')
            print('Falling back to OLS with country-wave FE...')

            m1b_ols = smf.ols(
                'anti_immig_index ~ task_z * C(welfare_regime, Treatment(reference="Nordic")) + agea_z + age_sq_z + female + eisced + hinctnta_z + urban + C(cntry_wave)',
                data=h1b_sample
            ).fit(cov_type='cluster', cov_kwds={'groups': h1b_sample['cntry_wave']})

            print('\nOLS Model 1b (FE) Results — key coefficients:')
            for term in ['task_z'] + [k for k in m1b_ols.params.index if 'task_z:' in k]:
                if term in m1b_ols.params.index:
                    coef = m1b_ols.params[term]
                    pval = m1b_ols.pvalues[term]
                    print(f'  {term}: {coef:.4f} (p={pval:.4f})')
                    log_finding(f'OLS 1b {term}: coef={coef:.4f}, p={pval:.4f}')

    # Model 2b: Redistribution ~ RTI x regime
    h2_vars_present = [v for v in ['task', 'redist_support', 'agea', 'age_sq', 'female',
                                    'eisced', 'hinctnta', 'urban', 'cntry_wave', 'cntry',
                                    'essround', 'welfare_regime'] if v in ess.columns]
    h2_sample = ess[h2_vars_present].dropna(subset=[v for v in h2_vars_present
                                                     if v not in ['cntry_wave', 'cntry', 'essround', 'welfare_regime']]).copy()
    h2b_sample = h2_sample[h2_sample['welfare_regime'].isin(
        ['Nordic', 'Continental', 'Liberal', 'Southern'])].copy()

    if len(h2b_sample) > 1000:
        for v in ['task', 'agea', 'hinctnta']:
            h2b_sample[f'{v}_z'] = (h2b_sample[v] - h2b_sample[v].mean()) / h2b_sample[v].std()
        h2b_sample['age_sq_z'] = h2b_sample['agea_z'] ** 2
        h2b_sample['welfare_regime'] = pd.Categorical(
            h2b_sample['welfare_regime'],
            categories=['Nordic', 'Continental', 'Liberal', 'Southern']
        )

        print('\nFitting Model 2b: Redistribution ~ RTI x Welfare regime (mixed model)...')
        try:
            m2b = smf.mixedlm(
                'redist_support ~ task_z * C(welfare_regime, Treatment(reference="Nordic")) + agea_z + age_sq_z + female + eisced + hinctnta_z + urban',
                data=h2b_sample,
                groups=h2b_sample['cntry_wave']
            ).fit(reml=True)

            print('\nModel 2b Results:')
            print(m2b.summary())

            interaction_terms = [k for k in m2b.params.index if 'task_z:' in k]
            print('\nKey interactions (vs. Nordic):')
            for term in interaction_terms:
                coef = m2b.params[term]
                pval = m2b.pvalues[term]
                print(f'  {term}: {coef:.4f} (p={pval:.4f})')
                log_finding(f'Model 2b {term}: coef={coef:.4f}, p={pval:.4f}')

        except Exception as e:
            print(f'Mixed model 2b failed: {e}')
            log_error('Model 2b', str(e))

    log_section('8. Diagnostic Models', 'SUCCESS')

except Exception as e:
    log_section('8. Diagnostic Models', 'FAILED', str(e))
    log_error('Models', traceback.format_exc())

# ============================================================
# SECTION 9: Autonomous Exploration
# ============================================================
print('\n' + '#'*60)
print('# SECTION 9: Autonomous Exploration')
print('#'*60)

# --- 9a. Education as moderator ---
try:
    plot_df_edu = ess[
        ess['welfare_regime'].isin(['Nordic', 'Continental', 'Liberal', 'Southern']) &
        ess['task'].notna() &
        ess['anti_immig_index'].notna() &
        ess['college'].notna()
    ].copy()

    if len(plot_df_edu) > 1000:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, (edu_val, edu_label) in zip(axes, [(0, 'Non-college'), (1, 'College')]):
            edf = plot_df_edu[plot_df_edu['college'] == edu_val]
            for regime in ['Nordic', 'Continental', 'Liberal', 'Southern']:
                rdf = edf[edf['welfare_regime'] == regime]
                if len(rdf) < 50:
                    continue
                valid = rdf[['task', 'anti_immig_index']].dropna()
                slope, intercept, r, p, se = stats.linregress(
                    valid['task'], valid['anti_immig_index'])
                color = regime_colors.get(regime, 'gray')
                ax.scatter([], [], c=color, label=f'{regime}: b={slope:.3f} (p={p:.3f})')
                # Plot binned means
                n_bins = min(15, max(5, len(rdf) // 80))
                rdf_copy = rdf.copy()
                rdf_copy['rti_bin'] = pd.qcut(rdf_copy['task'], q=n_bins, duplicates='drop')
                binned = rdf_copy.groupby('rti_bin', observed=True).agg(
                    rti_mean=('task', 'mean'),
                    dv_mean=('anti_immig_index', 'mean'),
                ).reset_index()
                ax.plot(binned['rti_mean'], binned['dv_mean'], 'o-', color=color,
                        markersize=4, alpha=0.7)

                log_finding(f'Education moderator: {edu_label} x {regime}: slope={slope:.4f}, p={p:.4f}')

            ax.set_xlabel('RTI Score')
            ax.set_ylabel('Anti-immigration index')
            ax.set_title(f'{edu_label} (N={len(edf):,})')
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(FIG_DIR / 'rti_by_education_regime.pdf', bbox_inches='tight')
        plt.savefig(FIG_DIR / 'rti_by_education_regime.png', bbox_inches='tight', dpi=150)
        plt.close()
        print('Saved: outputs/figures/rti_by_education_regime.pdf + .png')

except Exception as e:
    log_error('Education moderator', traceback.format_exc())

# --- 9b. Within-regime country variation ---
try:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for ax, regime in zip(axes.flat, ['Nordic', 'Continental', 'Liberal', 'Southern']):
        rdf = ess[(ess['welfare_regime'] == regime) &
                  ess['task'].notna() & ess['anti_immig_index'].notna()]
        countries = rdf['cntry'].unique()

        for cntry in sorted(countries):
            cdf = rdf[rdf['cntry'] == cntry]
            if len(cdf) < 50:
                continue
            valid = cdf[['task', 'anti_immig_index']].dropna()
            slope, intercept, r, p, se = stats.linregress(
                valid['task'], valid['anti_immig_index'])
            n_bins = min(10, max(3, len(cdf) // 100))
            cdf_copy = cdf.copy()
            cdf_copy['rti_bin'] = pd.qcut(cdf_copy['task'], q=n_bins, duplicates='drop')
            binned = cdf_copy.groupby('rti_bin', observed=True).agg(
                rti_mean=('task', 'mean'),
                dv_mean=('anti_immig_index', 'mean'),
            ).reset_index()
            ax.plot(binned['rti_mean'], binned['dv_mean'], 'o-', markersize=3,
                    alpha=0.6, label=f'{cntry} (b={slope:.3f})')

        ax.set_xlabel('RTI Score')
        ax.set_ylabel('Anti-immigration index')
        ax.set_title(f'{regime}')
        ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'within_regime_country_variation.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'within_regime_country_variation.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('Saved: outputs/figures/within_regime_country_variation.pdf + .png')

except Exception as e:
    log_error('Within-regime variation', traceback.format_exc())

# --- 9c. Deservingness in wave 8 ---
try:
    if 'narrow_deserving' in ess.columns:
        w8 = ess[ess['essround'] == 8].copy()
        w8_valid = w8[['task', 'narrow_deserving', 'anti_immig_index']].dropna()
        if len(w8_valid) > 100:
            corr = w8_valid[['task', 'narrow_deserving', 'anti_immig_index']].corr()
            print('\nWave 8 correlations (RTI, deservingness, anti-immigration):')
            print(corr.round(3))
            log_finding(f'Wave 8: RTI-deservingness corr = {corr.loc["task", "narrow_deserving"]:.3f}')
            log_finding(f'Wave 8: deservingness-anti_immig corr = {corr.loc["narrow_deserving", "anti_immig_index"]:.3f}')

except Exception as e:
    log_error('Deservingness analysis', traceback.format_exc())

# ============================================================
# SECTION 10: Save Master Dataset
# ============================================================
print('\n' + '#'*60)
print('# SECTION 10: Save Outputs')
print('#'*60)

try:
    # Save master dataset
    save_vars = ['essround', 'cntry', 'cntry_wave', 'welfare_regime', 'idno',
                 'isco08_raw', 'isco08_3d', 'task',
                 'imwbcnt', 'imueclt', 'imbgeco', 'anti_immig_index',
                 'gincdif', 'redist_support',
                 'trstprl', 'trstplt',
                 'agea', 'age_sq', 'female', 'eisced', 'eduyrs',
                 'hinctnta', 'urban', 'lrscale', 'college',
                 'fieldwork_year']

    # Add welfare indicators if they exist
    for wv in ['almp_pmp', 'unemp_pmp', 'socexp_t_pmp', 'active_passive_ratio',
               'training_share', 'incentive_share']:
        if wv in ess.columns:
            save_vars.append(wv)

    # Add deservingness if exists
    for dv in ['sbstrec', 'sbprvpv', 'sbbsntx', 'uentrjb', 'narrow_deserving']:
        if dv in ess.columns:
            save_vars.append(dv)

    save_vars = [v for v in save_vars if v in ess.columns]
    master = ess[save_vars].copy()
    master.to_csv(ANALYSIS_DIR / 'sorting_mechanism_master.csv', index=False)
    print(f'Saved: analysis/sorting_mechanism_master.csv ({len(master):,} rows, {len(save_vars)} cols)')

    report['sample']['final_master_rows'] = len(master)
    report['sample']['final_master_cols'] = len(save_vars)

    log_section('10. Save Master Dataset', 'SUCCESS')

except Exception as e:
    log_section('10. Save Master Dataset', 'FAILED', str(e))
    log_error('Save master', traceback.format_exc())

# --- Codebook ---
try:
    codebook_lines = ['# Codebook: sorting_mechanism_master.csv\n']
    codebook_lines.append(f'Generated: 2026-03-16\n')
    codebook_lines.append(f'Observations: {len(master):,}\n')
    codebook_lines.append(f'Variables: {len(save_vars)}\n\n')
    codebook_lines.append('| Variable | Type | N Valid | % Missing | Mean | SD | Min | Max | Description |\n')
    codebook_lines.append('|----------|------|--------|-----------|------|-----|-----|-----|-------------|\n')

    var_descriptions = {
        'essround': 'ESS wave number',
        'cntry': 'Country (ISO-2)',
        'cntry_wave': 'Country x wave identifier',
        'welfare_regime': 'Welfare regime classification',
        'idno': 'Respondent ID',
        'isco08_raw': 'Raw ISCO-08 code (4-digit)',
        'isco08_3d': 'Truncated ISCO-08 (3-digit)',
        'task': 'Routine task intensity score',
        'imwbcnt': 'Immigrants make country worse(0)-better(10)',
        'imueclt': 'Immigration undermines(0)-enriches(10) culture',
        'imbgeco': 'Immigration bad(0)-good(10) for economy',
        'anti_immig_index': 'Anti-immigration index (reversed, 0-10, higher=more anti)',
        'gincdif': 'Gov should reduce income differences (1-5)',
        'redist_support': 'Redistribution support (reversed, 1-5, higher=more support)',
        'trstprl': 'Trust in parliament (0-10)',
        'trstplt': 'Trust in politicians (0-10)',
        'agea': 'Age in years',
        'age_sq': 'Age squared',
        'female': 'Female (1=yes)',
        'eisced': 'Education (ISCED, 1-7)',
        'eduyrs': 'Years of education',
        'hinctnta': 'Household income decile (1-10)',
        'urban': 'Urban residence (1=big city/suburbs)',
        'lrscale': 'Left-right self-placement (0-10)',
        'college': 'College educated (ISCED>=6)',
        'fieldwork_year': 'ESS fieldwork year',
        'almp_pmp': 'ALMP spending (% GDP, CPDS)',
        'unemp_pmp': 'Unemployment benefits (% GDP, CPDS)',
        'socexp_t_pmp': 'Total social expenditure (% GDP, CPDS)',
        'active_passive_ratio': 'ALMP/(ALMP+unemployment) ratio',
        'training_share': 'Training share of ALMP',
        'incentive_share': 'Incentive share of ALMP',
        'sbstrec': 'Social benefits place too great a strain on economy (1-5, wave 8)',
        'sbprvpv': 'Social benefits prevent poverty (1-5, wave 8)',
        'sbbsntx': 'Social benefits cost businesses too much (1-5, wave 8)',
        'uentrjb': 'Most unemployed people do not really try to find a job (1-5, wave 8)',
        'narrow_deserving': 'Deservingness index (mean of raw sbstrec+uentrjb, higher=more INCLUSIVE; corrected 2026-06-14)',
    }

    for v in save_vars:
        col = master[v]
        n_valid = int(col.notna().sum())
        pct_miss = f'{(1 - col.notna().mean()) * 100:.1f}'
        dtype = str(col.dtype)
        desc = var_descriptions.get(v, '')

        if pd.api.types.is_numeric_dtype(col):
            mean_val = f'{col.mean():.2f}' if col.notna().any() else 'NA'
            sd_val = f'{col.std():.2f}' if col.notna().any() else 'NA'
            min_val = f'{col.min():.2f}' if col.notna().any() else 'NA'
            max_val = f'{col.max():.2f}' if col.notna().any() else 'NA'
        else:
            mean_val = sd_val = min_val = max_val = 'NA'

        codebook_lines.append(
            f'| {v} | {dtype} | {n_valid:,} | {pct_miss}% | {mean_val} | {sd_val} | {min_val} | {max_val} | {desc} |\n'
        )

    with open(ANALYSIS_DIR / 'codebook.md', 'w', encoding='utf-8') as f:
        f.writelines(codebook_lines)
    print('Saved: analysis/codebook.md')

except Exception as e:
    log_error('Codebook', traceback.format_exc())

# ============================================================
# SECTION 11: Write Overnight Report
# ============================================================
print('\n' + '#'*60)
print('# SECTION 11: Write Overnight Report')
print('#'*60)

report_lines = []
report_lines.append('# Overnight Report — Sorting Mechanism Analysis\n')
report_lines.append(f'**Date:** 2026-03-16\n')
report_lines.append(f'**Script:** `analysis/run_sorting_mechanism.py`\n\n')

report_lines.append('---\n\n')
report_lines.append('## 1. What Ran Successfully\n\n')
report_lines.append('| Section | Status | Detail |\n')
report_lines.append('|---------|--------|--------|\n')
for name, info in report['sections'].items():
    report_lines.append(f'| {name} | {info["status"]} | {info["detail"][:80]} |\n')

report_lines.append('\n---\n\n')
report_lines.append('## 2. What Failed and Why\n\n')
if report['errors']:
    for err in report['errors']:
        report_lines.append(f'### {err["section"]}\n')
        report_lines.append(f'**Error:** {err["error"][:200]}\n')
        if err.get('fix'):
            report_lines.append(f'**Attempted fix:** {err["fix"]}\n')
        report_lines.append('\n')
else:
    report_lines.append('No errors encountered.\n\n')

report_lines.append('---\n\n')
report_lines.append('## 3. Key Findings from Plots\n\n')
for finding in report['findings']:
    report_lines.append(f'- {finding}\n')

report_lines.append('\n### Plot A Interpretation\n\n')
report_lines.append('The binned scatter plots show the relationship between routine task intensity (RTI) ')
report_lines.append('and anti-immigration attitudes across welfare regime types. ')
report_lines.append('The key question is whether the slopes differ across regimes — ')
report_lines.append('steeper slopes in Liberal/conditional regimes would support H1.\n\n')

report_lines.append('### Plot B Interpretation\n\n')
report_lines.append('The redistribution plots show whether higher RTI predicts more or less ')
report_lines.append('support for redistribution, and whether this varies by regime. ')
report_lines.append('H2 predicts that universal/Nordic regimes channel vulnerability toward ')
report_lines.append('solidaristic attitudes rather than exclusionary ones.\n\n')

report_lines.append('---\n\n')
report_lines.append('## 4. Data Quality Issues\n\n')
report_lines.append(f'- **RTI match rate (among ISCO-coded):** {report["sample"].get("rti_match_conditional", "unknown")}\n')
report_lines.append(f'- **Total observations loaded:** {report["sample"].get("total_obs_loaded", "unknown"):,}\n')
report_lines.append(f'- **N with RTI scores:** {report["sample"].get("n_with_rti", "unknown"):,}\n')

# Country-wave coverage
cntry_wave_counts = ess.groupby('cntry')['essround'].nunique()
report_lines.append(f'- **Countries in sample:** {ess["cntry"].nunique()}\n')
report_lines.append(f'- **Waves used:** {sorted(ess["essround"].unique())}\n')
report_lines.append(f'- **Countries with all {len(primary_waves)} waves:** {(cntry_wave_counts == len(primary_waves)).sum()}\n')

report_lines.append('\n---\n\n')
report_lines.append('## 5. Sample Description\n\n')
report_lines.append(f'- **Final master dataset:** {report["sample"].get("final_master_rows", "?"):,} rows x {report["sample"].get("final_master_cols", "?")} variables\n')
report_lines.append(f'- **H1 analysis sample (RTI + anti-immig + controls):** {report["sample"].get("h1_sample", "?"):,}\n')
report_lines.append(f'- **Country coverage:** {sorted(ess["cntry"].unique())}\n')
report_lines.append(f'- **Wave coverage:** {sorted(ess["essround"].unique())}\n')

report_lines.append('\n### Observations by Regime\n\n')
regime_counts = ess['welfare_regime'].value_counts().sort_index()
for regime, count in regime_counts.items():
    report_lines.append(f'- {regime}: {count:,}\n')

report_lines.append('\n---\n\n')
report_lines.append('## 6. Decisions Still Needed\n\n')
report_lines.append('1. **Eastern European countries:** Included as a separate "Eastern" regime type. ')
report_lines.append('They comprise a large share of the ESS sample. Decide whether to include them ')
report_lines.append('in the main analysis (adds heterogeneity) or restrict to Western Europe (cleaner comparison).\n\n')
report_lines.append('2. **Continuous welfare indicators vs. regime categories:** CPDS data was merged. ')
report_lines.append('Check whether the continuous ALMP indicators perform better than the categorical regime variable.\n\n')
report_lines.append('3. **Radical right vote:** Not constructed in this run (party crosswalk merge is complex). ')
report_lines.append('This is needed for the full analysis but the attitude-based DVs are sufficient for initial exploration.\n\n')
report_lines.append('4. **Wave selection:** Currently using waves 6-9 (ISCO-08 available). ')
report_lines.append('Earlier waves use ISCO-88 and would need a crosswalk. Worth the effort?\n\n')

report_lines.append('---\n\n')
report_lines.append('## 7. Recommended Next Steps\n\n')
report_lines.append('1. **Review Plot A** — does the sorting pattern appear? Are the slopes different across regimes?\n')
report_lines.append('2. **Run the R multilevel models** — the Python mixed models are a diagnostic; ')
report_lines.append('the final specification should be in R with `lme4`/`lmerTest`.\n')
report_lines.append('3. **Construct radical right vote indicator** — merge Langenkamp crosswalk for binary DV.\n')
report_lines.append('4. **Download CWED universalism data** — this is the best continuous measure of welfare universalism.\n')
report_lines.append('5. **Consider adding waves 4-5** — ISCO-88 crosswalk is available in the repo.\n\n')

report_lines.append('---\n\n')
report_lines.append('## Files Produced\n\n')
report_lines.append('```\n')
report_lines.append('analysis/\n')
report_lines.append('  sorting_mechanism_master.csv    — merged analysis dataset\n')
report_lines.append('  codebook.md                     — variable descriptions\n')
report_lines.append('  overnight_report.md             — this file\n')
report_lines.append('  run_sorting_mechanism.py        — standalone pipeline script\n')
report_lines.append('\noutputs/figures/\n')
report_lines.append('  fig2_rti_vs_antiimmig_by_regime.pdf + .png\n')
report_lines.append('  fig4_rti_vs_redistribution_by_regime.pdf + .png\n')
report_lines.append('  welfare_vs_rti_slopes.pdf + .png\n')
report_lines.append('  rti_distribution.pdf + .png\n')
report_lines.append('  rti_by_education_regime.pdf + .png\n')
report_lines.append('  within_regime_country_variation.pdf + .png\n')
report_lines.append('\noutputs/tables/\n')
report_lines.append('  summary_stats.csv\n')
report_lines.append('```\n')

with open(ANALYSIS_DIR / 'overnight_report.md', 'w', encoding='utf-8') as f:
    f.writelines(report_lines)
print('Saved: analysis/overnight_report.md')

print('\n' + '='*60)
print('  PIPELINE COMPLETE')
print('='*60)
print(f'  Master dataset: {report["sample"].get("final_master_rows", "?"):,} rows')
print(f'  All plots saved to outputs/figures/')
print(f'  Report at analysis/overnight_report.md')
print('='*60)
