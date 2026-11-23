"""Patient-level statistics reported in the paper.

Every estimate is the mean of per-patient means; intervals are percentile
bootstrap intervals over patients (5,000 resamples); tests are Wilcoxon
signed-rank tests on patient means.

Usage: python run_patient_level_stats.py <results dir>
Writes <results dir>/patient_level_stats.json
"""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

D = sys.argv[1]
B = 5000
KEY = ['file', 'label', 'row_top', 'x', 'w', 'h']
LABELS = {'cartilage': 'Cartilage - Partial Thickness loss/defect',
          'meniscus': 'Meniscus Tear',
          'mcl': 'Ligament - MCL Low-Mod Grade Sprain'}
rng = np.random.default_rng(2027)


def summarise(df, col):
    """Mean of patient means, patient bootstrap CI, Wilcoxon on patient means."""
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[col])
    pm = df.groupby('file')[col].mean().values
    out = dict(mean=float(pm.mean()), n_boxes=int(len(df)), n_patients=int(len(pm)),
               n_pos=int((pm > 0).sum()), n_neg=int((pm < 0).sum()))
    if len(pm) >= 3:
        boots = [rng.choice(pm, len(pm)).mean() for _ in range(B)]
        out['ci_lo'], out['ci_hi'] = map(float, np.percentile(boots, [2.5, 97.5]))
    if len(pm) >= 5:
        out['wilcoxon_p'] = float(wilcoxon(pm).pvalue)
    return out


def load(R):
    if R == 4:
        reg = pd.read_csv(f"{D}/regions_R4.csv")
        pla = pd.read_csv(f"{D}/placebo_R4.csv")
        ren = {}
    else:
        reg = pd.read_csv(f"{D}/accel_regions_R{R}.csv")
        pla = pd.read_csv(f"{D}/accel_placebo_R{R}.csv")
        ren = {'ssim_les': 'ssim_les_cg', 'ssim_ctl': 'ssim_sizectl_cg'}
    reg = reg.rename(columns=ren); pla = pla.rename(columns=ren)
    pla['dist'] = (pla.slice - pla.src_slice).abs()
    return reg, pla


def same_position(reg, pla, col, lo, hi):
    sp = (pla[(pla.dist >= lo) & (pla.dist <= hi)]
          .replace([np.inf, -np.inf], np.nan)
          .groupby(KEY + ['src_slice'])[col].mean().rename('sp').reset_index())
    m = reg.merge(sp, left_on=KEY + ['slice'], right_on=KEY + ['src_slice'])
    m['eff'] = m[col] - m.sp
    return m


out = {'estimator': 'mean of patient means; patient bootstrap 95% CI; Wilcoxon on patient means'}
reg, pla = load(4)
SPS = {'sp_all': (1, 99), 'sp_d3': (1, 3), 'sp_d1': (1, 1)}

# Table 1 (and PSNR equivalents)
t1 = {}
for rec in ['cg', 'zf']:
    for met in ['ssim', 'psnr']:
        les = f'{met}_les_{rec}'
        reg['sm'] = reg[les] - reg[f'{met}_sizectl_{rec}']
        reg['am'] = reg[les] - reg[f'{met}_appctl_{rec}']
        row = {'sm': summarise(reg, 'sm'), 'am': summarise(reg, 'am')}
        for k, (lo, hi) in SPS.items():
            row[k] = summarise(same_position(reg, pla, les, lo, hi), 'eff')
        t1[f'{rec}_{met}'] = row
out['table1'] = t1

# Table 2: per finding, CG-SENSE SSIM
t2 = {}
reg['sm'] = reg.ssim_les_cg - reg.ssim_sizectl_cg
reg['am'] = reg.ssim_les_cg - reg.ssim_appctl_cg
for name, lab in LABELS.items():
    sub = reg[reg.label == lab]
    row = {'sm': summarise(sub, 'sm'), 'am': summarise(sub, 'am')}
    for k, (lo, hi) in SPS.items():
        m = same_position(reg, pla, 'ssim_les_cg', lo, hi)
        row[k] = summarise(m[m.label == lab], 'eff')
    m = same_position(reg, pla, 'psnr_les_cg', 1, 99)
    row['sp_all_psnr'] = summarise(m[m.label == lab], 'eff')
    t2[name] = row
out['table2'] = t2

# Distance curves (Figure 3b) and acceleration
bins = {'1': (1, 1), '2': (2, 2), '3': (3, 3), '4-5': (4, 5), '6-9': (6, 9), '10+': (10, 99)}
dist, accel = {}, {}
for R in [2, 4, 8]:
    r, p = load(R)
    dist[R] = {}
    for grp in ['all', 'cartilage', 'meniscus']:
        dist[R][grp] = {}
        for b, (lo, hi) in bins.items():
            m = same_position(r, p, 'ssim_les_cg', lo, hi)
            if grp != 'all':
                m = m[m.label == LABELS[grp]]
            if len(m):
                dist[R][grp][b] = summarise(m, 'eff')
    r['sm'] = r.ssim_les_cg - r.ssim_sizectl_cg
    accel[R] = {'sm': summarise(r, 'sm')}
    for k, (lo, hi) in SPS.items():
        accel[R][k] = summarise(same_position(r, p, 'ssim_les_cg', lo, hi), 'eff')
out['distance'] = dist
out['acceleration'] = accel

# Adjacent-slice subset description
d1 = same_position(reg, pla, 'ssim_les_cg', 1, 1)
ann = reg[['file', 'slice']].drop_duplicates()
annotated = set(map(tuple, ann.values))
edge = [((f, s - 1) in annotated) or ((f, s + 1) in annotated) for f, s in zip(d1.file, d1.slice)]
out['adjacent_subset'] = dict(n_boxes=int(len(d1)), n_patients=int(d1.file.nunique()),
                              n_edge_slices=int(sum(edge)),
                              median_area_subset=float(d1.area.median()),
                              median_area_all=float(reg.area.median()))

# Lesion-free noise floor: same box on adjacent unannotated slice pairs
nf = {}
for rec in ['cg', 'zf']:
    col = f'ssim_les_{rec}'
    rows = []
    for k, g in pla.groupby(KEY + ['src_slice']):
        s = g.set_index('slice')[col]
        for sl in s.index:
            if sl + 1 in s.index:
                rows.append(abs(s[sl + 1] - s[sl]))
    a = np.array(rows)
    nf[rec] = dict(n_pairs=int(len(a)), median_abs=float(np.median(a)),
                   mean_abs=float(a.mean()), q95_abs=float(np.quantile(a, 0.95)))
    m = same_position(reg, pla, col, 1, 1)
    nf[rec]['lesion_adjacent_median_abs'] = float(m.eff.abs().median())
    nf[rec]['lesion_adjacent_q95_abs'] = float(m.eff.abs().quantile(0.95))
out['noise_floor'] = nf


# Comparator disagreement: difference of a box's effect under two comparators
cmp_ = {}
for rec in ['cg', 'zf']:
    col = f'ssim_les_{rec}'
    m = same_position(reg, pla, col, 1, 99)
    m['sm_minus_sp'] = (m[col] - m[f'ssim_sizectl_{rec}']) - m.eff
    m['am_minus_sp'] = (m[col] - m[f'ssim_appctl_{rec}']) - m.eff
    cmp_[rec] = {'sm_minus_sp_all': summarise(m, 'sm_minus_sp'),
                 'am_minus_sp_all': summarise(m, 'am_minus_sp')}
out['comparator_difference'] = cmp_

# Same box, far versus near same-position comparators
def far_minus_near(sub_reg, near, far):
    a = same_position(sub_reg, pla, 'ssim_les_cg', *near)[KEY + ['slice', 'eff']].rename(columns={'eff': 'near'})
    b = same_position(sub_reg, pla, 'ssim_les_cg', *far)[KEY + ['slice', 'eff']].rename(columns={'eff': 'far'})
    j = a.merge(b, on=KEY + ['slice'])
    j['far_minus_near'] = j.far - j.near
    return summarise(j, 'far_minus_near')
out['within_box'] = {
    'all_far4_minus_adjacent': far_minus_near(reg, (1, 1), (4, 99)),
    'cartilage_far6_minus_near3': far_minus_near(reg[reg.label == LABELS['cartilage']], (1, 3), (6, 99)),
}

json.dump(out, open(f"{D}/patient_level_stats.json", 'w'), indent=1)


def fmt(s):
    ci = f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}]" if 'ci_lo' in s else '--'
    p = f"{s['wilcoxon_p']:.3f}" if 'wilcoxon_p' in s else '--'
    return f"{s['mean']:+.3f} {ci} {s['n_pos']}/{s['n_neg']} p={p} (boxes {s['n_boxes']}, pts {s['n_patients']})"


for k, row in t1.items():
    print('==', k)
    for c, s in row.items():
        print(f"   {c:7s} {fmt(s)}")
for k, row in t2.items():
    print('==', k)
    for c, s in row.items():
        print(f"   {c:12s} {fmt(s)}")
for R in accel:
    print('== accel R', R)
    for c, s in accel[R].items():
        print(f"   {c:7s} {fmt(s)}")
for R in dist:
    for grp in dist[R]:
        print('== dist R', R, grp, {b: round(s['mean'], 3) for b, s in dist[R][grp].items()})
print(json.dumps(out['adjacent_subset']), json.dumps(nf))
for k, v in out['comparator_difference'].items():
    for c, s_ in v.items():
        print('== comparator difference', k, c, fmt(s_))
for k, s_ in out['within_box'].items():
    print('== within box', k, fmt(s_))
