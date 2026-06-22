"""Two faces of solidarity -- confirmatory pipeline (S0-S5).

Frozen spec: ../ANALYSIS_PLAN.md. This script reproduces the ground-truth gates,
estimates the redistribution-vs-deservingness dissociation under RTI, decomposes
the deservingness battery item-by-item (with the sbprvpv sign-flip coherence
check), runs robustness, and inspects the latent factor structure.

ESTIMATORS
  S1 ground-truth gate     : statsmodels mixedlm random-intercept (matches the
                             shipped house spec; hits model7 step1 = -0.029657).
  S2/S3 deservingness arms : OLS + country fixed effects (via demeaning) +
                             cluster-robust SE at country-wave (= country in
                             wave 8, 23 clusters) + wild-cluster bootstrap-t
                             (Webb 6-point AND Rademacher 2-point) for the
                             few-clusters regime.

CODING (pinned to ground truth, not to names; see ANALYSIS_PLAN.md)
  All battery items are raw ESS 1=agree..5=disagree.
  narrow_deserving == mean(sbstrec, uentrjb) EXACTLY (asserted).
  Higher = MORE INCLUSIVE for uentrjb, sbstrec, sbbsntx, narrow_deserving (raw).
  sbprvpv is the one PRO-welfare-worded item -> reverse (6 - x) so higher=inclusive
  ONLY inside the deserving_inclusive index. The raw sbprvpv is kept for the
  coherence check (it must flip sign vs the 4 negatively-worded items).

Run:  python pipeline.py
Writes: ../results/results.txt and ../results/numbers.csv
"""

import sys
import io
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# UTF-8 stdout (Danish chars, arrows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
RESULTS = PROJ / 'results'
RESULTS.mkdir(exist_ok=True)

# Restricted ESS-derived data is NOT included in this repo (see ../README.md).
# Point TWO_FACES_DATA at a directory holding the master file + ground-truth references to run.
DATA_DIR = Path(os.environ.get('TWO_FACES_DATA', str(PROJ / 'data')))
MASTER = DATA_DIR / 'sorting_mechanism_master.csv'

REGIME = ['Nordic', 'Continental', 'Liberal', 'Southern', 'Eastern']
CONTROLS = ['agea', 'age_sq', 'female', 'college', 'hinctnta', 'urban']

# Reproducibility for the bootstrap
N_BOOT = 9999
BOOT_SEED = 20260614
RADEMACHER2 = np.array([-1.0, 1.0])
WEBB6 = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5),
                  np.sqrt(0.5), 1.0, np.sqrt(1.5)])

# ---------------------------------------------------------------------------
# Output capture
# ---------------------------------------------------------------------------
_lines = []
_numbers = []  # rows of dict(name, value, se, p, n, spec)


def say(*args):
    msg = ' '.join(str(a) for a in args)
    print(msg)
    _lines.append(msg)


def hr(title=''):
    say('')
    say('=' * 78)
    if title:
        say(title)
        say('=' * 78)


def num(name, value, se=None, p=None, n=None, spec=''):
    """Record a citable number into numbers.csv."""
    _numbers.append({
        'name': name,
        'value': '' if value is None else f'{value:.6g}',
        'se': '' if se is None else f'{se:.6g}',
        'p': '' if p is None else f'{p:.3e}',
        'n': '' if n is None else int(n),
        'spec': spec,
    })


# ===========================================================================
# Data load + construction
# ===========================================================================
def read_csv_danish(path, **kw):
    try:
        return pd.read_csv(path, encoding='utf-8-sig', **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='latin-1', **kw)


def load_data():
    say(f'Loading master: {MASTER}')
    df = read_csv_danish(MASTER, low_memory=False)
    say(f'  shape: {df.shape[0]:,} x {df.shape[1]}')

    # --- task_z: standardize raw `task` over the FULL master, exactly as the
    #     original pipeline did (the parent analysis pipeline). ---
    tmean, tstd = df['task'].mean(), df['task'].std()
    df['task_z'] = (df['task'] - tmean) / tstd
    say(f'  task_z = (task - {tmean:.6f}) / {tstd:.6f}  [std over full master, '
        f'n_task={df["task"].notna().sum():,}]')

    # --- identity check: narrow_deserving == mean(sbstrec, uentrjb) ---
    nd_recon = df[['sbstrec', 'uentrjb']].mean(axis=1, skipna=True)
    nd_recon[df[['sbstrec', 'uentrjb']].notna().sum(axis=1) < 1] = np.nan
    both = df['narrow_deserving'].notna() & nd_recon.notna()
    max_diff = (df.loc[both, 'narrow_deserving'] - nd_recon[both]).abs().max()
    say(f'  IDENTITY narrow_deserving == mean(sbstrec, uentrjb): max|diff| = {max_diff:.8f}')
    assert max_diff < 1e-9, 'narrow_deserving identity FAILED'

    # --- oriented items: higher = MORE INCLUSIVE ---
    # uentrjb, sbstrec, sbbsntx raw already higher=inclusive (1=agree-restrictive..5=disagree)
    # sbprvpv is pro-welfare-worded -> reverse (6 - x) so higher=inclusive
    df['sbprvpv_rev'] = 6 - df['sbprvpv']
    incl_items = ['uentrjb', 'sbstrec', 'sbbsntx', 'sbprvpv_rev']
    df['deserving_inclusive'] = df[incl_items].mean(axis=1, skipna=True)
    df.loc[df[incl_items].notna().sum(axis=1) < 1, 'deserving_inclusive'] = np.nan

    return df


# ===========================================================================
# S1 -- reproduce ground-truth gates
# ===========================================================================
def build_antiimmig_analysis(df):
    """The shipped `analysis` frame (the parent paper's analysis frame):
    dropna on anti_immig_index + task_z + welfare_regime + cntry_wave + controls,
    restricted to the 5 welfare regimes. This is the frame model7 lives in."""
    base = ['anti_immig_index', 'task_z', 'welfare_regime', 'cntry_wave'] + CONTROLS
    a = df.dropna(subset=base).copy()
    a = a[a['welfare_regime'].isin(REGIME)].copy()
    return a


def s1_gates(df):
    hr('S1 -- GROUND-TRUTH REPRODUCTION GATES')
    analysis = build_antiimmig_analysis(df)
    say(f'  shipped anti-immig analysis frame: n={len(analysis):,}, '
        f'clusters={analysis["cntry_wave"].nunique()}')

    # --- Gate 1: model7 step1 (mixedlm RE, wave 8, anti_immig-conditioned) ---
    w8 = analysis[(analysis['essround'] == 8) & analysis['narrow_deserving'].notna()].copy()
    f7 = 'narrow_deserving ~ task_z + agea + age_sq + female + college + hinctnta + urban'
    m7 = smf.mixedlm(f7, data=w8, groups=w8['cntry_wave']).fit(reml=True)
    c, se, p, n = (m7.params['task_z'], m7.bse['task_z'], m7.pvalues['task_z'], int(m7.nobs))
    say('')
    say('  GATE 1 -- model7 step1 (narrow_deserving ~ task_z + controls, mixedlm RE, wave 8):')
    say(f'    task_z coef = {c:.6f}   se = {se:.6f}   p = {p:.3e}   n = {n}')
    say(f'    TARGET      = -0.029657              0.005615        1.281e-07     28860')
    gate1 = abs(c - (-0.029657)) < 1e-4 and n == 28860
    say(f'    GATE 1 PASS = {gate1}')
    num('S1_model7_step1_RTI_narrow_deserving', c, se, p, n,
        'mixedlm RE groups=cntry_wave, wave8, anti_immig-conditioned (GROUND TRUTH)')

    # --- Gate 2: paper RTI -> redist_support (all-wave) ---
    ar = analysis.dropna(subset=['redist_support']).copy()
    f5 = ('redist_support ~ task_z * C(welfare_regime, Treatment(reference="Nordic"))'
          ' + agea + age_sq + female + college + hinctnta + urban')
    # random-intercept (matches the parent paper's model 5 = 0.04397)
    m5ri = smf.mixedlm(f5, data=ar, groups=ar['cntry_wave']).fit(reml=True)
    cri, seri, pri, nri = (m5ri.params['task_z'], m5ri.bse['task_z'],
                           m5ri.pvalues['task_z'], int(m5ri.nobs))
    # random-slopes on task_z (matches the paper's headline 0.041)
    try:
        m5rs = smf.mixedlm(f5, data=ar, groups=ar['cntry_wave'],
                           re_formula='~task_z').fit(reml=True)
        crs, sers, prs, nrs = (m5rs.params['task_z'], m5rs.bse['task_z'],
                               m5rs.pvalues['task_z'], int(m5rs.nobs))
    except Exception as e:
        crs = sers = prs = None
        nrs = nri
        say(f'    [random-slopes M5 failed: {e}]')
    say('')
    say('  GATE 2 -- RTI -> redist_support (paper +0.041), all-wave M5 mixedlm:')
    say(f'    random-INTERCEPT task_z = {cri:.6f}  se={seri:.6f}  p={pri:.3e}  n={nri}'
        f'   (parent paper model 5 = 0.04397)')
    if crs is not None:
        say(f'    random-SLOPES   task_z = {crs:.6f}  se={sers:.6f}  p={prs:.3e}  n={nrs}'
            f'   (paper headline 0.04105)')
    gate2 = abs(cri - 0.04397) < 1e-3
    say(f'    GATE 2 PASS (RI matches 0.044) = {gate2}')
    num('S1_M5_RTI_redist_randomintercept', cri, seri, pri, nri,
        'mixedlm RI all-wave (matches parent paper model 5 = 0.04397)')
    if crs is not None:
        num('S1_M5_RTI_redist_randomslopes', crs, sers, prs, nrs,
            'mixedlm random-slopes on task_z all-wave (paper headline 0.041)')

    say('')
    say(f'  S1 BOTH GATES PASS = {gate1 and gate2}')
    return gate1, gate2


# ===========================================================================
# FE + cluster-robust + wild bootstrap toolkit (standard few-clusters approach)
# ===========================================================================
def _demean(df, cols, group):
    out = df.copy()
    for c in cols:
        out[c] = out[c] - out.groupby(group)[c].transform('mean')
    return out


def fe_ols(df, y, xs, fe_group='cntry', cluster='cntry'):
    """OLS with `fe_group` fixed effects via within (demeaning), cluster-robust
    SE clustered on `cluster`. In wave 8, cntry_wave == cntry, so country FE +
    country clustering = country-wave clustering (23 clusters)."""
    use = df.dropna(subset=[y] + xs + [fe_group, cluster]).copy()
    dm = _demean(use, [y] + xs, group=fe_group)
    Y = dm[y].values
    X = dm[xs].values
    model = sm.OLS(Y, X)
    res = model.fit(cov_type='cluster', cov_kwds={'groups': use[cluster].values})
    return {
        'use': use, 'y': y, 'xs': xs, 'fe_group': fe_group, 'cluster': cluster,
        'beta': dict(zip(xs, res.params)),
        'se': dict(zip(xs, res.bse)),
        't': dict(zip(xs, res.tvalues)),
        'p': dict(zip(xs, res.pvalues)),
        'n_obs': int(len(use)),
        'n_clusters': int(use[cluster].nunique()),
        'res': res,
    }


def wild_cluster_bootstrap_t(df, y, xs, target, fe_group='cntry', cluster='cntry',
                             n_boot=N_BOOT, seed=BOOT_SEED, weights=WEBB6):
    """Restricted wild-cluster bootstrap-t (Cameron-Gelbach-Miller 2008) for
    H0: beta_target == 0. FE absorbed by demeaning on fe_group; clustering on
    `cluster`. Returns p_boot and t_obs."""
    use = df.dropna(subset=[y] + xs + [fe_group, cluster]).copy()
    dm = _demean(use, [y] + xs, group=fe_group)
    Y = dm[y].values.astype(float)
    X = dm[xs].values.astype(float)
    groups = use[cluster].values
    uniq = np.unique(groups)
    G = len(uniq)
    cidx = {g: np.where(groups == g)[0] for g in uniq}

    def crve_t(Yv, Xv, target_idx):
        XtX = Xv.T @ Xv
        XtX_inv = np.linalg.pinv(XtX)
        beta = XtX_inv @ (Xv.T @ Yv)
        resid = Yv - Xv @ beta
        meat = np.zeros_like(XtX)
        for g in uniq:
            ix = cidx[g]
            sg = Xv[ix].T @ resid[ix]
            meat += np.outer(sg, sg)
        c = G / (G - 1)
        V = c * (XtX_inv @ meat @ XtX_inv)
        b = beta[target_idx]
        se = np.sqrt(V[target_idx, target_idx])
        return b, se, (b / se if se > 0 else np.nan), beta, resid

    ti = xs.index(target)
    b_obs, se_obs, t_obs, _, _ = crve_t(Y, X, ti)
    # restricted residuals: impose beta_target = 0
    keep = [i for i in range(len(xs)) if i != ti]
    Xr = X[:, keep]
    beta_r = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ Y)
    resid_r = Y - Xr @ beta_r
    fitted_r = Y - resid_r

    rng = np.random.default_rng(seed)
    count = used = 0
    for _ in range(n_boot):
        w = rng.choice(weights, size=G)
        wmap = {g: w[i] for i, g in enumerate(uniq)}
        wvec = np.array([wmap[g] for g in groups])
        Ystar = fitted_r + resid_r * wvec
        _, _, t_star, _, _ = crve_t(Ystar, X, ti)
        if np.isnan(t_star):
            continue
        used += 1
        if abs(t_star) >= abs(t_obs):
            count += 1
    p = (count + 1) / (used + 1) if used else np.nan
    return {'p_boot': p, 't_obs': float(t_obs), 'b_obs': float(b_obs),
            'se_obs': float(se_obs), 'n_clusters': G}


# ===========================================================================
# S2/S3 joint wave-8 frame (NO welfare_regime filter -- S2 has no regime term;
# country FE absorbs country. Keeping the filter drops IL+RU -> 21 clusters.
# Dropping it -> 23 clusters, the plan's check.)
# ===========================================================================
def build_joint_w8(df, outcomes):
    w8 = df[df['essround'] == 8].copy()
    need = ['task_z'] + CONTROLS + outcomes
    j = w8.dropna(subset=need).copy()
    return j


def s2_dissociation(df):
    hr('S2 -- PRIMARY DISSOCIATION (the headline)')
    joint = build_joint_w8(df, ['redist_support', 'deserving_inclusive'])
    say(f'  joint wave-8 frame (no regime filter): n={len(joint):,}, '
        f'clusters(cntry)={joint["cntry"].nunique()}   [plan check: 23]')
    say(f'  countries: {sorted(joint["cntry"].unique())}')

    xs = ['task_z'] + CONTROLS
    results = {}
    for outcome, expect in [('redist_support', 'POSITIVE (>0)'),
                            ('deserving_inclusive', 'NEGATIVE (<0)')]:
        fit = fe_ols(joint, outcome, xs, fe_group='cntry', cluster='cntry')
        b = fit['beta']['task_z']
        se = fit['se']['task_z']
        p = fit['p']['task_z']
        say('')
        say(f'  {outcome} ~ task_z + controls  (OLS + country FE, cluster-robust @ country)')
        say(f'    task_z beta = {b:.6f}   se = {se:.6f}   p_analytic = {p:.3e}'
            f'   n = {fit["n_obs"]}   clusters = {fit["n_clusters"]}')
        say(f'    expected sign: {expect}')
        # wild bootstrap (Webb + Rademacher)
        bw = wild_cluster_bootstrap_t(joint, outcome, xs, 'task_z',
                                      fe_group='cntry', cluster='cntry', weights=WEBB6)
        br = wild_cluster_bootstrap_t(joint, outcome, xs, 'task_z',
                                      fe_group='cntry', cluster='cntry', weights=RADEMACHER2)
        say(f'    wild-bootstrap p: Webb6 = {bw["p_boot"]:.4f}   '
            f'Rademacher2 = {br["p_boot"]:.4f}   (t_obs = {bw["t_obs"]:.3f}, '
            f'{bw["n_clusters"]} clusters, {N_BOOT} reps)')
        results[outcome] = dict(b=b, se=se, p=p, p_webb=bw['p_boot'],
                                p_rad=br['p_boot'], n=fit['n_obs'], G=fit['n_clusters'])
        num(f'S2_RTI_{outcome}', b, se, p, fit['n_obs'],
            'OLS+countryFE, cluster-robust @ country (wave8)')
        num(f'S2_RTI_{outcome}_wildboot_webb', b, None, bw['p_boot'], fit['n_obs'],
            'wild-cluster bootstrap-t Webb6, H0 beta=0')
        num(f'S2_RTI_{outcome}_wildboot_rademacher', b, None, br['p_boot'], fit['n_obs'],
            'wild-cluster bootstrap-t Rademacher2, H0 beta=0')

    rd = results['redist_support']
    di = results['deserving_inclusive']
    # dissociation: redist > 0 AND deserving_inclusive < 0, both distinguishable
    redist_pos_sig = rd['b'] > 0 and rd['p_webb'] < 0.05
    deserv_neg_sig = di['b'] < 0 and di['p_webb'] < 0.05
    dissociation = redist_pos_sig and deserv_neg_sig
    say('')
    say('  DISSOCIATION TEST (both must hold, both distinguishable from 0 by wild bootstrap):')
    say(f'    RTI -> redist_support > 0 and sig (Webb p<.05): {redist_pos_sig} '
        f'(beta={rd["b"]:+.4f}, p_webb={rd["p_webb"]:.4f})')
    say(f'    RTI -> deserving_inclusive < 0 and sig (Webb p<.05): {deserv_neg_sig} '
        f'(beta={di["b"]:+.4f}, p_webb={di["p_webb"]:.4f})')
    say(f'    DISSOCIATION HOLDS = {dissociation}')
    num('S2_dissociation_holds', 1 if dissociation else 0, None, None, None,
        'redist>0&sig AND deserving_inclusive<0&sig (wild boot Webb p<.05)')
    return results, dissociation


# ===========================================================================
# S3 -- battery decomposition + sbprvpv coherence check
# ===========================================================================
def s3_battery(df):
    hr('S3 -- BATTERY DECOMPOSITION + sbprvpv COHERENCE CHECK')

    # Coherence check is on the RAW items. 4 negatively-worded items
    # (uentrjb, sbstrec, sbbsntx, narrow_deserving) must take NEGATIVE task_z
    # coefficients (higher RTI -> lower raw value -> more restrictive); the one
    # PRO-worded item (sbprvpv, raw) must FLIP POSITIVE if it is a uniform signal.
    raw_items = ['uentrjb', 'sbstrec', 'sbbsntx', 'narrow_deserving', 'sbprvpv']
    xs = ['task_z'] + CONTROLS
    signs = {}
    say('  Item-by-item RTI coefficients (RAW coding, OLS + country FE, cluster-robust):')
    say('  (raw 1=agree..5=disagree; for the 4 neg-worded items higher=inclusive,')
    say('   so a NEGATIVE task_z coef = RTI pushes toward restrictive)')
    say('')
    for item in raw_items:
        joint = build_joint_w8(df, [item])
        fit = fe_ols(joint, item, xs, fe_group='cntry', cluster='cntry')
        b, se, p = fit['beta']['task_z'], fit['se']['task_z'], fit['p']['task_z']
        bw = wild_cluster_bootstrap_t(joint, item, xs, 'task_z',
                                      fe_group='cntry', cluster='cntry', weights=WEBB6)
        signs[item] = np.sign(b)
        flag = ''
        if item == 'sbprvpv':
            flag = '  <-- PRO-worded: should FLIP POSITIVE if uniform signal'
        say(f'    {item:18s} beta={b:+.6f}  se={se:.6f}  p={p:.3e}  '
            f'p_webb={bw["p_boot"]:.4f}  n={fit["n_obs"]}{flag}')
        num(f'S3_RTI_{item}_raw', b, se, p, fit['n_obs'],
            'OLS+countryFE cluster-robust, RAW item (wave8)')
        num(f'S3_RTI_{item}_raw_wildboot_webb', b, None, bw['p_boot'], fit['n_obs'],
            'wild bootstrap Webb6 RAW item')

    # coherence verdict
    neg_items = ['uentrjb', 'sbstrec', 'sbbsntx', 'narrow_deserving']
    four_negative = all(signs[i] < 0 for i in neg_items)
    sbprvpv_positive = signs['sbprvpv'] > 0
    coherence = four_negative and sbprvpv_positive
    say('')
    say('  COHERENCE CHECK (sbprvpv sign-flip):')
    for i in neg_items:
        say(f'    {i:18s} sign = {int(signs[i]):+d}  (expect - )')
    say(f'    {"sbprvpv":18s} sign = {int(signs["sbprvpv"]):+d}  (expect + : the flip)')
    say(f'    4 neg-worded items all NEGATIVE = {four_negative}')
    say(f'    sbprvpv FLIPS POSITIVE          = {sbprvpv_positive}')
    say(f'    COHERENCE (uniform restrictive signal) HOLDS = {coherence}')
    say('    Interpretation: flip holding => uniform "RTI -> more restrictive"')
    say('                    signal across the welfare battery (the discriminator).')
    num('S3_coherence_four_negative', 1 if four_negative else 0, None, None, None,
        '4 neg-worded items all have negative RTI coef')
    num('S3_coherence_sbprvpv_flips_positive', 1 if sbprvpv_positive else 0, None, None, None,
        'sbprvpv raw RTI coef is positive (the flip)')
    num('S3_coherence_holds', 1 if coherence else 0, None, None, None,
        'sbprvpv coherence check passes')
    return signs, coherence


# ===========================================================================
# S4 -- robustness
# ===========================================================================
def s4_robustness(df):
    hr('S4 -- ROBUSTNESS')

    # (a) OLS+FE vs mixed model, both arms
    say('  (a) Estimator: OLS+countryFE+cluster vs mixedlm RE (cntry_wave), wave 8')
    for outcome in ['redist_support', 'deserving_inclusive']:
        joint = build_joint_w8(df, [outcome])
        xs = ['task_z'] + CONTROLS
        fe = fe_ols(joint, outcome, xs, fe_group='cntry', cluster='cntry')
        f = (f'{outcome} ~ task_z + agea + age_sq + female + college + hinctnta + urban')
        mm = smf.mixedlm(f, data=joint, groups=joint['cntry_wave']).fit(reml=True)
        say(f'    {outcome:20s} OLS+FE beta={fe["beta"]["task_z"]:+.6f} '
            f'(p={fe["p"]["task_z"]:.2e})   mixedlm beta={mm.params["task_z"]:+.6f} '
            f'(p={mm.pvalues["task_z"]:.2e})')
        num(f'S4a_{outcome}_mixedlm', mm.params['task_z'], mm.bse['task_z'],
            mm.pvalues['task_z'], int(mm.nobs), 'mixedlm RE cntry_wave wave8 (robustness)')

    # (b) controls in vs out -- does the deservingness arm survive income+education?
    say('')
    say('  (b) Controls in/out (does the deservingness arm survive class confounders?)')
    specs = {
        'nocontrols': ([], 'no controls'),
        'demographics': (['agea', 'age_sq', 'female'], 'demographics only (age,age2,female)'),
        'college': (['agea', 'age_sq', 'female', 'college'], '+ college'),
        'incomefull': (CONTROLS, '+ income (full)'),
    }
    for outcome in ['redist_support', 'deserving_inclusive']:
        say(f'    -- {outcome} --')
        for key, (ctrls, label) in specs.items():
            joint = df[df['essround'] == 8].dropna(subset=['task_z', outcome] + ctrls).copy()
            xs = ['task_z'] + ctrls if ctrls else ['task_z']
            fit = fe_ols(joint, outcome, xs, fe_group='cntry', cluster='cntry')
            bw = wild_cluster_bootstrap_t(joint, outcome, xs, 'task_z',
                                          fe_group='cntry', cluster='cntry', weights=WEBB6)
            say(f'      {label:42s} beta={fit["beta"]["task_z"]:+.6f}  '
                f'p={fit["p"]["task_z"]:.3e}  p_webb={bw["p_boot"]:.4f}  n={fit["n_obs"]}')
            num(f'S4b_{outcome}_{key}', fit['beta']['task_z'],
                fit['se']['task_z'], fit['p']['task_z'], fit['n_obs'],
                f'controls={label}; OLS+countryFE cluster')
            num(f'S4b_{outcome}_{key}_wildboot_webb', fit['beta']['task_z'],
                None, bw['p_boot'], fit['n_obs'],
                f'controls={label}; wild bootstrap Webb6')

    # (c) ordinal (ologit) for single-item outcomes -- scale compression
    say('')
    say('  (c) Ordinal (ologit) for single-item outcomes (addresses scale compression)')
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    for outcome in ['redist_support', 'uentrjb', 'sbstrec', 'sbbsntx', 'sbprvpv']:
        joint = build_joint_w8(df, [outcome])
        # ologit with country dummies as FE, cluster-robust on cntry
        try:
            Xcols = ['task_z'] + CONTROLS
            cdum = pd.get_dummies(joint['cntry'], prefix='c', drop_first=True).astype(float)
            Xo = pd.concat([joint[Xcols].reset_index(drop=True),
                            cdum.reset_index(drop=True)], axis=1)
            yo = joint[outcome].astype(int).reset_index(drop=True)
            om = OrderedModel(yo, Xo, distr='logit').fit(method='bfgs', disp=0,
                                                         maxiter=200)
            b, se, p = om.params['task_z'], om.bse['task_z'], om.pvalues['task_z']
            say(f'    {outcome:18s} ologit task_z beta={b:+.6f}  se={se:.6f}  '
                f'p={p:.3e}  n={int(om.nobs)}')
            num(f'S4c_ologit_{outcome}', b, se, p, int(om.nobs),
                'ordered logit + country dummies (scale-compression robustness)')
        except Exception as e:
            say(f'    {outcome:18s} ologit failed: {e}')

    # (d) standardized effect sizes (z-score the outcomes) so the two arms compare
    say('')
    say('  (d) Standardized effect sizes (outcome z-scored; betas in SD units)')
    for outcome in ['redist_support', 'deserving_inclusive']:
        joint = build_joint_w8(df, [outcome]).copy()
        joint[outcome + '_z'] = ((joint[outcome] - joint[outcome].mean())
                                 / joint[outcome].std())
        xs = ['task_z'] + CONTROLS
        fit = fe_ols(joint, outcome + '_z', xs, fe_group='cntry', cluster='cntry')
        b, se, p = fit['beta']['task_z'], fit['se']['task_z'], fit['p']['task_z']
        say(f'    {outcome:20s} std beta = {b:+.6f} SD per 1 SD RTI  '
            f'(se={se:.6f}, p={p:.3e}, n={fit["n_obs"]})')
        num(f'S4d_std_{outcome}', b, se, p, fit['n_obs'],
            'outcome z-scored; OLS+countryFE; beta in SD units')


# ===========================================================================
# S5 -- one latent dimension?
# ===========================================================================
def s5_factor(df):
    hr('S5 -- IS IT ONE LATENT DIMENSION?')
    # all oriented so higher=inclusive/pro: redist_support, uentrjb, sbstrec,
    # sbbsntx, sbprvpv_rev
    cols = ['redist_support', 'uentrjb', 'sbstrec', 'sbbsntx', 'sbprvpv_rev']
    w8 = df[df['essround'] == 8].dropna(subset=cols).copy()
    say(f'  wave-8 complete-case n = {len(w8):,} on {cols}')

    # correlation matrix
    corr = w8[cols].corr()
    say('')
    say('  Correlation matrix (all oriented higher=inclusive/pro-redistribution):')
    say('    ' + '  '.join(f'{c[:8]:>10s}' for c in cols))
    for i, ri in enumerate(cols):
        row = '  '.join(f'{corr.loc[ri, cj]:>10.3f}' for cj in cols)
        say(f'    {ri[:18]:18s} {row}')
    # redist vs deservingness block correlation
    deserv = ['uentrjb', 'sbstrec', 'sbbsntx', 'sbprvpv_rev']
    rd_corrs = [corr.loc['redist_support', d] for d in deserv]
    say(f'\n  redist_support x deservingness items: '
        f'mean r = {np.mean(rd_corrs):.3f}  (range {min(rd_corrs):.3f}..{max(rd_corrs):.3f})')
    within_deserv = [corr.loc[a, b] for i, a in enumerate(deserv) for b in deserv[i+1:]]
    say(f'  within-deservingness mean r = {np.mean(within_deserv):.3f}')
    num('S5_corr_redist_deserving_mean', np.mean(rd_corrs), spec='mean r(redist, deserving items)')
    num('S5_corr_within_deserving_mean', np.mean(within_deserv), spec='mean r within deservingness items')

    # Cronbach alpha for deserving_inclusive (4 items)
    def cronbach(dfm):
        k = dfm.shape[1]
        item_var = dfm.var(axis=0, ddof=1).sum()
        tot_var = dfm.sum(axis=1).var(ddof=1)
        return (k / (k - 1)) * (1 - item_var / tot_var)
    a_di = cronbach(w8[deserv])
    say(f'\n  Cronbach alpha, deserving_inclusive (4 oriented items) = {a_di:.4f}')
    num('S5_cronbach_deserving_inclusive', a_di, n=len(w8),
        spec='4 oriented items uentrjb,sbstrec,sbbsntx,sbprvpv_rev')

    # PCA / factor: eigenvalues of correlation matrix; do redist + deserving split?
    eigvals, eigvecs = np.linalg.eigh(corr.values)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    say('\n  PCA on the 5-item correlation matrix (all oriented same way):')
    say(f'    eigenvalues: {np.round(eigvals, 3).tolist()}')
    say(f'    variance explained: {np.round(eigvals / eigvals.sum() * 100, 1).tolist()} %')
    n_above1 = int((eigvals > 1).sum())
    say(f'    factors with eigenvalue > 1 (Kaiser): {n_above1}')
    num('S5_n_factors_kaiser', n_above1, spec='eigenvalues>1 on 5-item corr matrix')
    num('S5_eigenvalue_1', eigvals[0], spec='largest eigenvalue (5-item)')
    num('S5_eigenvalue_2', eigvals[1], spec='2nd eigenvalue (5-item)')

    # loadings on first 2 components
    say('\n  Loadings (first 2 components, sqrt(eigval)*eigvec):')
    L1 = eigvecs[:, 0] * np.sqrt(eigvals[0])
    L2 = eigvecs[:, 1] * np.sqrt(eigvals[1])
    say(f'    {"item":18s} {"PC1":>8s} {"PC2":>8s}')
    for i, c in enumerate(cols):
        say(f'    {c:18s} {L1[i]:>8.3f} {L2[i]:>8.3f}')
    say('')
    if n_above1 >= 2:
        say('    => >1 factor with eigenvalue>1: supports a multi-dimensional structure')
        say('       (redist "how much" vs deservingness "for whom" may load separately).')
    else:
        say('    => single dominant factor (eigenvalue>1): the "two faces" framing')
        say('       is weakened -- redist and deservingness share one latent dimension.')


# ===========================================================================
# Main
# ===========================================================================
def main():
    say('TWO FACES OF SOLIDARITY -- confirmatory pipeline')
    say(f'  python {sys.version.split()[0]}  pandas {pd.__version__}  '
        f'numpy {np.__version__}  statsmodels {statsmodels.__version__}  '
        f'scipy {scipy.__version__}')
    say(f'  N_BOOT={N_BOOT}  seed={BOOT_SEED}')

    df = load_data()

    g1, g2 = s1_gates(df)
    if not (g1 and g2):
        say('\n!!! S1 GATE FAILED -- numbers below are NOT trustworthy. Investigate. !!!')

    s2_res, dissociation = s2_dissociation(df)
    signs, coherence = s3_battery(df)
    s4_robustness(df)
    s5_factor(df)

    # ---- Summary ----
    hr('SUMMARY')
    say(f'  S1 gate 1 (model7 step1)    : {g1}')
    say(f'  S1 gate 2 (paper +0.041)    : {g2}')
    say(f'  S2 dissociation holds       : {dissociation}')
    say(f'  S3 sbprvpv coherence holds  : {coherence}')

    num('SUMMARY_S1_gate1', 1 if g1 else 0, spec='model7 step1 reproduced')
    num('SUMMARY_S1_gate2', 1 if g2 else 0, spec='paper RTI->redist reproduced')
    num('SUMMARY_S2_dissociation', 1 if dissociation else 0, spec='dissociation holds')
    num('SUMMARY_S3_coherence', 1 if coherence else 0, spec='sbprvpv coherence holds')

    # ---- write outputs ----
    (RESULTS / 'results.txt').write_text('\n'.join(_lines), encoding='utf-8')
    pd.DataFrame(_numbers).to_csv(RESULTS / 'numbers.csv', index=False, encoding='utf-8-sig')
    say(f'\n  wrote {RESULTS / "results.txt"}')
    say(f'  wrote {RESULTS / "numbers.csv"}  ({len(_numbers)} numbers)')


if __name__ == '__main__':
    main()
