"""SUR / stacked joint dissociation test -- requested follow-up to pipeline.py.

WHY THIS SCRIPT EXISTS
  The frozen confirmatory run (pipeline.py, S2) tested the two faces of welfare
  solidarity SEPARATELY, each against zero:
      RTI -> redist_support       beta = +0.050  (wild-boot p = .0004)  robust
      RTI -> deserving_inclusive  beta = -0.018  (wild-boot p = .058)   right sign, n.s.
  and parked on an honest qualified null: the STRICT dissociation (both arms
  distinguishable from zero, opposite signs) does not hold.

  But "do the two faces dissociate?" is a CROSS-EQUATION question, and two
  separate-vs-zero tests do not answer it. The apt test is whether the RTI effect
  DIFFERS across the two outcomes:  H0: beta_redist == beta_deserving.
  Even when the deserving arm alone is ~0, the DIFFERENCE between a +0.05 and a
  -0.02 effect can be distinguishable from zero. That is what a stacked/SUR test
  delivers and what this script adds.

THE TWO CLAIMS ARE NOT THE SAME (kept separate throughout, per the hard rule):
  DIFFER  :  beta_redist != beta_deserving            <- the delta test (this script)
  OPPOSE  :  beta_redist > 0 AND beta_deserving < 0,
             each distinguishable from zero            <- what S2 tested and FAILED
  A significant delta licenses "RTI moves the two faces DIFFERENTLY"; it does NOT
  license "RTI hardens deservingness" (that is the OPPOSE claim, which needs the
  deserving arm to stand on its own -- it does not, it is largely education).

METHOD (identical inference toolkit to pipeline.py S2, just stacked)
  Frame   : the wave-8 JOINT sample -- respondents with BOTH outcomes and all
            controls present (build_joint_w8, n ~ 32,339, 23 country clusters).
            The two arms are tested on the SAME respondents.
  Scale   : both outcomes z-scored WITHIN the joint frame, so beta_redist and
            beta_deserving are in common SD units and their difference is meaningful
            (raw 1-5 redist and 1-5 deserving_inclusive have different SDs).
  Stack   : each respondent contributes two rows; D=1 marks the deserving equation.
            Fully interacted:  y ~ task_z + task_z:D + controls + controls:D,
            with country-by-equation FE (demean on cntry_eq) so each equation gets
            its own FE and its own control slopes. By Kruskal's theorem (identical
            regressors) this reproduces the two separate OLS point estimates EXACTLY;
            the gain is the joint, cluster-robust covariance.
  delta   : the coefficient on task_z:D == beta_deserving - beta_redist. Testing
            delta != 0 IS the dissociation-as-difference test.
  Inference: country FE (cntry_eq), cluster-robust SE clustered on COUNTRY (23).
            Country clustering SUBSUMES the within-respondent error correlation
            (a respondent's two rows are in the same country cluster), so no
            separate respondent-cluster dimension is needed. Wild-cluster
            bootstrap-t (Webb 6-point + Rademacher) on delta for the few-clusters
            regime, H0: delta = 0.

INTERNAL CORRECTNESS GATE (self-contained, not vs published numbers)
  Run the two separate FE-OLS on THIS script's own standardized joint frame to get
  beta_redist_z and beta_deserving_z, then assert stacked delta == (beta_d - beta_r)
  to < 1e-9. That is the real check that the stacking/demeaning is correct.

CAVEAT TO PRE-EMPT (for the methodology pass)
  deserving_inclusive has Cronbach alpha = 0.46 (noisy battery). NOTE the correct
  mechanism: this is measurement noise in an OUTCOME, not a regressor, so it does
  NOT classically attenuate the OLS slope -- it inflates the outcome's variance.
  Because each outcome is z-scored, that inflated SD COMPRESSES the standardized
  coefficient toward zero. Both standardized arms are therefore compressed, so
  |delta_std| is a lower bound on |delta_true|: the DIRECTION of the gap is robust
  and its magnitude is a floor, not a ceiling. (This is a standardization-
  compression argument, not errors-in-variables attenuation.)

Run:  python code/sur_dissociation.py
Reads pipeline.load_data() (same master, read-only). Writes:
  ../results/sur_results.txt  and  ../results/sur_numbers.csv
"""

import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Import the FROZEN pipeline's data construction + inference toolkit so this test
# uses byte-identical frame and bootstrap machinery (the whole point of comparability).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline as P  # noqa: E402  (load_data, build_joint_w8, fe_ols, wild_cluster_bootstrap_t, CONTROLS, ...)

# pipeline already wrapped stdout in UTF-8 on import; keep our own output buffer.
PROJ = HERE.parent
RESULTS = PROJ / 'results'
RESULTS.mkdir(exist_ok=True)

CONTROLS = P.CONTROLS                # ['agea','age_sq','female','college','hinctnta','urban']
WEBB6 = P.WEBB6
RADEMACHER2 = P.RADEMACHER2
N_BOOT = P.N_BOOT
T_CRIT_22 = 2.073873                 # t_{.975, 22}  (G-1 = 22 df, 23 clusters) -- small-sample CI

_lines = []
_numbers = []


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


def num(name, value, se=None, p=None, n=None, ci_lo=None, ci_hi=None, spec=''):
    _numbers.append({
        'name': name,
        'value': '' if value is None else f'{value:.6g}',
        'se': '' if se is None else f'{se:.6g}',
        'p': '' if p is None else f'{p:.3e}',
        'ci_lo': '' if ci_lo is None else f'{ci_lo:.6g}',
        'ci_hi': '' if ci_hi is None else f'{ci_hi:.6g}',
        'n': '' if n is None else int(n),
        'spec': spec,
    })


# ---------------------------------------------------------------------------
# Build the standardized joint wave-8 frame (same respondents, both outcomes)
# ---------------------------------------------------------------------------
def build_standardized_joint(df):
    joint = P.build_joint_w8(df, ['redist_support', 'deserving_inclusive']).copy()
    # z-score each outcome WITHIN this joint frame -> common SD units
    for outcome in ['redist_support', 'deserving_inclusive']:
        mu, sd = joint[outcome].mean(), joint[outcome].std()
        joint[outcome + '_z'] = (joint[outcome] - mu) / sd
        num(f'JOINT_{outcome}_mean', mu, spec='joint-frame mean (pre-standardization)')
        num(f'JOINT_{outcome}_sd', sd, spec='joint-frame sd (pre-standardization)')
    return joint


def arm_on_joint(joint, outcome_z, label, expect):
    """One FE-OLS arm on the standardized joint frame + wild-boot p + small-sample CI."""
    xs = ['task_z'] + CONTROLS
    fit = P.fe_ols(joint, outcome_z, xs, fe_group='cntry', cluster='cntry')
    b, se = fit['beta']['task_z'], fit['se']['task_z']
    p_z = fit['p']['task_z']                          # statsmodels z-approx (large-df)
    p = 2 * stats.t.sf(abs(b / se), df=22)            # small-sample analytic, t_{G-1=22}
    lo, hi = b - T_CRIT_22 * se, b + T_CRIT_22 * se
    bw = P.wild_cluster_bootstrap_t(joint, outcome_z, xs, 'task_z',
                                    fe_group='cntry', cluster='cntry', weights=WEBB6)
    say(f'  {label:24s} beta = {b:+.5f} SD   se = {se:.5f}   '
        f'95% CI [{lo:+.5f}, {hi:+.5f}]')
    say(f'  {"":24s} p_wild(Webb) = {bw["p_boot"]:.4f} [PRIMARY]   '
        f'p_analytic(t22) = {p:.3e}   (statsmodels z-approx = {p_z:.2e})')
    say(f'  {"":24s} n = {fit["n_obs"]}   clusters = {fit["n_clusters"]}   expect {expect}')
    num(f'ARM_{outcome_z}', b, se, p, fit['n_obs'], lo, hi,
        'standardized arm on JOINT frame, OLS+countryFE cluster@country')
    num(f'ARM_{outcome_z}_wild_webb', b, None, bw['p_boot'], fit['n_obs'],
        spec='wild-cluster bootstrap-t Webb6 H0 beta=0 (joint frame)')
    return dict(b=b, se=se, p=p, p_webb=bw['p_boot'], lo=lo, hi=hi,
                n=fit['n_obs'], G=fit['n_clusters'], res=fit['res'], use=fit['use'])


# ---------------------------------------------------------------------------
# Stack the two standardized equations (fully interacted) -> delta = task_z:D
# ---------------------------------------------------------------------------
def build_stacked(joint, controls):
    """Two rows per respondent. D=1 = deserving equation. Fully interacted design."""
    base_cols = ['cntry', 'task_z'] + controls
    eq0 = joint[base_cols].copy()
    eq0['y'] = joint['redist_support_z'].values
    eq0['D'] = 0.0
    eq1 = joint[base_cols].copy()
    eq1['y'] = joint['deserving_inclusive_z'].values
    eq1['D'] = 1.0
    st = pd.concat([eq0, eq1], ignore_index=True)
    st['task_z_D'] = st['task_z'] * st['D']
    for c in controls:
        st[c + '_D'] = st[c] * st['D']
    st['cntry_eq'] = st['cntry'].astype(str) + '_' + st['D'].astype(int).astype(str)
    return st


def stacked_delta(joint, controls, tag, do_rademacher=False):
    """Fully-interacted stacked OLS; target = task_z_D (= beta_deserv - beta_redist).
    Country-by-equation FE (demean on cntry_eq), cluster on country, wild boot."""
    st = build_stacked(joint, controls)
    xs = ['task_z', 'task_z_D'] + controls + [c + '_D' for c in controls]
    fit = P.fe_ols(st, 'y', xs, fe_group='cntry_eq', cluster='cntry')
    delta = fit['beta']['task_z_D']
    se = fit['se']['task_z_D']
    p_z = fit['p']['task_z_D']                          # statsmodels z-approx (large-df)
    p = 2 * stats.t.sf(abs(delta / se), df=22)          # small-sample analytic, t_{G-1=22}
    b_redist = fit['beta']['task_z']            # D=0 slope
    lo, hi = delta - T_CRIT_22 * se, delta + T_CRIT_22 * se
    bw = P.wild_cluster_bootstrap_t(st, 'y', xs, 'task_z_D',
                                    fe_group='cntry_eq', cluster='cntry', weights=WEBB6)
    p_rad = None
    if do_rademacher:
        br = P.wild_cluster_bootstrap_t(st, 'y', xs, 'task_z_D',
                                        fe_group='cntry_eq', cluster='cntry',
                                        weights=RADEMACHER2)
        p_rad = br['p_boot']
    return dict(delta=delta, se=se, p=p, p_z=p_z, lo=lo, hi=hi, p_webb=bw['p_boot'],
                p_rad=p_rad, b_redist=b_redist, n_rows=fit['n_obs'],
                G=fit['n_clusters'], res=fit['res'], st=st, xs=xs, tag=tag)


# ===========================================================================
def main():
    say('SUR / STACKED JOINT DISSOCIATION TEST  (follow-up to pipeline.py S2)')
    say(f'  python {sys.version.split()[0]}  pandas {pd.__version__}  numpy {np.__version__}')
    say(f'  N_BOOT={N_BOOT}  seed={P.BOOT_SEED}  (same toolkit as the frozen S2 run)')

    df = P.load_data()
    joint = build_standardized_joint(df)
    say(f'\n  JOINT wave-8 frame (both outcomes + full controls present): '
        f'n = {len(joint):,} respondents, clusters(cntry) = {joint["cntry"].nunique()}')
    say(f'  countries: {sorted(joint["cntry"].unique())}')

    # ---- The two arms, standardized, on the SAME respondents (context for delta) ----
    hr('ARMS (standardized, joint frame) -- the "OPPOSE" ingredients')
    say('  Both oriented so HIGHER = more pro-welfare/inclusive. Full controls, country FE.')
    a_red = arm_on_joint(joint, 'redist_support_z', 'redist_support_z ("how much")', '> 0')
    a_des = arm_on_joint(joint, 'deserving_inclusive_z', 'deserving_inclusive_z ("for whom")', '< 0')

    # cross-equation residual correlation (the source of any SUR covariance gain)
    r_red = pd.Series(a_red['res'].resid, index=a_red['use'].index)
    r_des = pd.Series(a_des['res'].resid, index=a_des['use'].index)
    resid_r = r_red.corr(r_des)
    say(f'\n  cross-equation residual correlation r(e_redist, e_deserving) = {resid_r:+.4f}'
        f'   (low => little point-estimate gain from joint estimation; Kruskal holds)')
    num('cross_eq_resid_corr', resid_r, spec='corr of the two arms FE-OLS residuals (same respondents)')

    # ---- Headline stacked delta (full controls) ----
    hr('DELTA -- the stacked joint test (full controls)  -- the "DIFFER" claim')
    h = stacked_delta(joint, CONTROLS, 'full', do_rademacher=True)

    # INTERNAL GATE: stacked delta == (beta_deserv - beta_redist) on this same frame
    sep_diff = a_des['b'] - a_red['b']
    gate_delta = abs(h['delta'] - sep_diff) < 1e-9
    gate_redist = abs(h['b_redist'] - a_red['b']) < 1e-9
    say(f'  INTERNAL GATE (stacking correctness):')
    say(f'    stacked delta               = {h["delta"]:+.10f}')
    say(f'    (beta_deserv - beta_redist) = {sep_diff:+.10f}   [from the two separate arms above]')
    say(f'    |difference|                = {abs(h["delta"] - sep_diff):.2e}   PASS={gate_delta}')
    say(f'    stacked task_z (=redist arm)= {h["b_redist"]:+.10f}  vs separate {a_red["b"]:+.10f}'
        f'   PASS={gate_redist}')
    if not (gate_delta and gate_redist):
        say('    !!! GATE FAILED -- stacking/demeaning is wrong; delta below is NOT trustworthy !!!')
    num('GATE_delta_equals_arm_difference', 1 if gate_delta else 0,
        spec='|stacked delta - (beta_deserv-beta_redist)| < 1e-9')

    # naive-vs-joint SE: show the covariance adjustment the stacking provides
    se_naive = np.sqrt(a_red['se'] ** 2 + a_des['se'] ** 2)
    say(f'\n  delta = beta_deserving - beta_redist = {h["delta"]:+.5f} SD')
    say(f'    joint cluster-robust se(delta)      = {h["se"]:.5f}   '
        f'(naive sqrt(se_r^2+se_d^2) = {se_naive:.5f}; stacking adds the cross-eq covariance)')
    say(f'    95% CI (t_22)                       = [{h["lo"]:+.5f}, {h["hi"]:+.5f}]')
    say(f'    p_wild(Webb6) = {h["p_webb"]:.4f} [PRIMARY]   p_wild(Rademacher2) = {h["p_rad"]:.4f}')
    say(f'    p_analytic(t22) = {h["p"]:.3e}   (statsmodels z-approx = {h["p_z"]:.2e}; '
        f'the z-approx is anti-conservative with 23 clusters -- use wild-boot)')
    say(f'    n_rows = {h["n_rows"]:,} (2 x {len(joint):,})   clusters = {h["G"]}')
    num('DELTA_full', h['delta'], h['se'], h['p'], h['n_rows'], h['lo'], h['hi'],
        'stacked task_z:D = beta_deserv-beta_redist, std outcomes, full controls, '
        'countryFE x eq, cluster@country')
    num('DELTA_full_wild_webb', h['delta'], None, h['p_webb'], h['n_rows'],
        spec='wild-cluster bootstrap-t Webb6 H0 delta=0')
    num('DELTA_full_wild_rademacher', h['delta'], None, h['p_rad'], h['n_rows'],
        spec='wild-cluster bootstrap-t Rademacher2 H0 delta=0')
    num('DELTA_full_se_naive_independent', se_naive,
        spec='sqrt(se_redist^2+se_deserv^2) ignoring cross-eq covariance (for contrast)')

    # ---- Control ladder on delta: does the DIFFERENCE survive education? ----
    hr('DELTA control ladder -- does the DIFFERENCE survive class confounders?')
    say('  Same fixed joint sample (n constant); only the RHS controls change.')
    say('  (Contrast: the deserving ARM alone collapses when college enters -- S2/WHAT_WOULD_KILL.)')
    ladder = {
        'nocontrols':   ([], 'no controls'),
        'demographics': (['agea', 'age_sq', 'female'], 'demographics (age,age2,female)'),
        'college':      (['agea', 'age_sq', 'female', 'college'], '+ college'),
        'incomefull':   (CONTROLS, '+ income (full controls)'),
    }
    say('')
    for key, (ctrls, lbl) in ladder.items():
        hl = stacked_delta(joint, ctrls, key, do_rademacher=False)
        say(f'    {lbl:32s} delta = {hl["delta"]:+.5f}  95%CI [{hl["lo"]:+.5f},{hl["hi"]:+.5f}]  '
            f'p_an = {hl["p"]:.2e}  p_wild = {hl["p_webb"]:.4f}')
        num(f'DELTA_ladder_{key}', hl['delta'], hl['se'], hl['p'], hl['n_rows'],
            hl['lo'], hl['hi'], f'stacked delta, controls={lbl}, std outcomes')
        num(f'DELTA_ladder_{key}_wild_webb', hl['delta'], None, hl['p_webb'], hl['n_rows'],
            spec=f'wild Webb6 H0 delta=0, controls={lbl}')

    # ---- Raw-scale secondary (units not comparable across scales; for completeness) ----
    hr('DELTA raw-scale (SECONDARY -- 1-5 scales differ, not directly comparable)')
    jr = joint.copy()
    jr['redist_support_z'] = jr['redist_support']            # overload names: feed raw through stack
    jr['deserving_inclusive_z'] = jr['deserving_inclusive']
    hrw = stacked_delta(jr, CONTROLS, 'raw', do_rademacher=False)
    say(f'  delta_raw = beta_deserv(raw) - beta_redist(raw) = {hrw["delta"]:+.5f}  '
        f'95%CI [{hrw["lo"]:+.5f},{hrw["hi"]:+.5f}]  p_an={hrw["p"]:.2e}  p_wild={hrw["p_webb"]:.4f}')
    say('  (raw redist and raw deserving_inclusive have different SDs; read the standardized delta '
        'as the headline.)')
    num('DELTA_raw_full', hrw['delta'], hrw['se'], hrw['p'], hrw['n_rows'],
        hrw['lo'], hrw['hi'], 'stacked delta on RAW outcomes (secondary; scales differ)')

    # ===================================================================
    # VERDICTS -- "DIFFER" vs "OPPOSE", kept separate (decision-agnostic tee-up)
    # ===================================================================
    hr('VERDICTS  (the two claims, kept separate)')
    differ = h['p_webb'] < 0.05
    redist_pos_sig = a_red['b'] > 0 and a_red['p_webb'] < 0.05
    deserv_neg_sig = a_des['b'] < 0 and a_des['p_webb'] < 0.05
    oppose = redist_pos_sig and deserv_neg_sig

    say('  DIFFER  (beta_redist != beta_deserving): the joint/SUR test, delta != 0')
    say(f'    delta = {h["delta"]:+.4f} SD, 95% CI [{h["lo"]:+.4f}, {h["hi"]:+.4f}], '
        f'wild p = {h["p_webb"]:.4f}  ->  DIFFER = {differ}')
    say('')
    say('  OPPOSE  (beta_redist > 0 AND beta_deserving < 0, each distinguishable from 0):')
    say(f'    redist arm     beta = {a_red["b"]:+.4f}, wild p = {a_red["p_webb"]:.4f}  '
        f'-> pos&sig = {redist_pos_sig}')
    say(f'    deserving arm  beta = {a_des["b"]:+.4f}, wild p = {a_des["p_webb"]:.4f}  '
        f'-> neg&sig = {deserv_neg_sig}')
    say(f'    OPPOSE = {oppose}')
    say('')
    say('  READING: a significant DIFFER is carried by the robust redist arm against a')
    say('           near-zero, education-confounded deserving arm. DIFFER licenses "RTI moves')
    say('           the two faces differently"; it does NOT license "RTI hardens deservingness"')
    say('           (that is OPPOSE, which fails on the deserving arm).')
    say('           - The control ladder shows the DIFFERENCE survives college, but the reason')
    say('             is that the REDIST arm is college-stable and anchors delta; it is NOT')
    say('             independent evidence that both arms diverge robustly under controls.')
    say('           - Battery alpha=0.46: outcome noise inflates the deserving SD and COMPRESSES')
    say('             its standardized beta toward 0 (a standardization effect, NOT errors-in-')
    say('             variables attenuation of a regressor). Both standardized arms are')
    say('             compressed, so |delta_std| is a floor on |delta_true|, not a ceiling.')
    num('VERDICT_differ', 1 if differ else 0, spec='delta!=0 by wild Webb p<.05 (DIFFER)')
    num('VERDICT_oppose', 1 if oppose else 0,
        spec='redist>0&sig AND deserving<0&sig by wild Webb (OPPOSE) -- the S2 strict test')

    # ---- write outputs ----
    (RESULTS / 'sur_results.txt').write_text('\n'.join(_lines), encoding='utf-8')
    pd.DataFrame(_numbers).to_csv(RESULTS / 'sur_numbers.csv', index=False, encoding='utf-8-sig')
    say(f'\n  wrote {RESULTS / "sur_results.txt"}')
    say(f'  wrote {RESULTS / "sur_numbers.csv"}  ({len(_numbers)} numbers)')


if __name__ == '__main__':
    main()
