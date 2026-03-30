
import itertools
import math
import re
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# reuse previous helpers
spec_ob = importlib.util.spec_from_file_location('ob', '/mnt/data/overlay_bridge_and_shell_tests.py')
ob = importlib.util.module_from_spec(spec_ob)
spec_ob.loader.exec_module(ob)

spec_pvn = importlib.util.spec_from_file_location('pvn', '/mnt/data/phase_vs_nullmodels.py')
pvn = importlib.util.module_from_spec(spec_pvn)
spec_pvn.loader.exec_module(pvn)

OUT = Path('/mnt/data')
BETAS = [0.0, 1.0, 1.5, 2.0, 5.0, 10.0]
GEOMETRIES = pvn.GEOMETRIES
CRITERIA = pvn.CRITERIA

def support_degree(fam: str) -> int:
    m = re.match(r'(\d+)-axis', fam)
    return int(m.group(1)) if m else 0

def overlay_field(u, pair, pts_per_cell=5, target_sep=8, pad=10):
    G0 = np.abs(ob.dense_xyz_carrier(u))
    base = np.zeros(tuple(np.array(G0.shape) + 2 * pad), dtype=float)
    base[pad:pad + G0.shape[0], pad:pad + G0.shape[1], pad:pad + G0.shape[2]] = G0
    off = np.array([pad, pad, pad], dtype=float)

    p1, p2 = [(ob.cell_center(idx, pts_per_cell) - 1) + off for idx in pair]
    center = np.array(base.shape) // 2
    t1 = center + np.array([-target_sep // 2, 0, 0], dtype=float)
    t2 = center + np.array([ target_sep // 2, 0, 0], dtype=float)
    s1 = t1 - p1
    s2 = t2 - p2

    O = ob.shift(base, s1, order=1, mode='constant', cval=0.0, prefilter=False)
    O += ob.shift(base, s2, order=1, mode='constant', cval=0.0, prefilter=False)
    return O, center.astype(int), t1.astype(int), t2.astype(int)

def longest_true_run(mask):
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best

def bridge_metrics(O, center, t1, t2):
    c = center
    xs = np.arange(min(t1[0], t2[0]), max(t1[0], t2[0]) + 1)
    line = O[xs, c[1], c[2]]
    off_line = 0.25 * (
        O[xs, c[1] + 2, c[2]] + O[xs, c[1] - 2, c[2]] +
        O[xs, c[1], c[2] + 2] + O[xs, c[1], c[2] - 2]
    )
    lobe1 = float(O[tuple(t1)])
    lobe2 = float(O[tuple(t2)])
    lobe_mean = 0.5 * (lobe1 + lobe2) + 1e-12
    interior = line[1:-1]
    thr = 0.45 * min(lobe1, lobe2)
    neck_floor = float(np.min(interior) / lobe_mean)
    neck_mid_ratio = float(O[tuple(c)] / lobe_mean)
    axiality_ratio = float(np.mean(line) / (np.mean(off_line) + 1e-12))
    continuity_frac = float(np.mean(interior > thr))
    bridge_span_ratio = float(longest_true_run(interior > thr) / max(len(interior), 1))
    if len(interior) >= 3:
        sec = np.diff(interior, n=2)
        straightness = float(1.0 / (1.0 + np.mean(np.abs(sec)) / (np.mean(interior) + 1e-12)))
    else:
        straightness = 0.0
    return {
        'neck_floor': neck_floor,
        'neck_mid_ratio': neck_mid_ratio,
        'axiality_ratio': axiality_ratio,
        'continuity_frac': continuity_frac,
        'bridge_span_ratio': bridge_span_ratio,
        'straightness': straightness,
    }

def compare(real, controls):
    controls = np.asarray(controls, dtype=float)
    mu = float(np.mean(controls))
    sd = float(np.std(controls, ddof=0))
    z = float((real - mu) / (sd + 1e-12))
    exceed = float(np.mean(real > controls))
    return mu, sd, z, exceed

# Step D
full_rows = []
case_rows = []

for geom, lengths in GEOMETRIES:
    for beta in BETAS:
        rows = ob.solve_mode_full(lengths, beta, modes=10)
        winners = ob.pick_winners(rows)
        for crit, win in winners.items():
            xyz = win['xyz']
            best = ob.best_opposite_pair(xyz)
            shell = best['target_shell']
            target_pair = best['target_pair']
            O, center, t1, t2 = overlay_field(win['u'], target_pair)
            target_metrics = bridge_metrics(O, center, t1, t2)

            controls = []
            for pair in ob.NONOP_PAIRS[shell]:
                # same shell different angle
                Oi, ci, t1i, t2i = overlay_field(win['u'], pair)
                controls.append(bridge_metrics(Oi, ci, t1i, t2i))

            row = {
                'geometry': geom, 'beta': beta, 'criterion': crit,
                'winner_q': win['dominant_q'], 'winner_family': win['q_family'],
                'target_shell': shell, 'target_pair': str(target_pair),
                'pair_strength': best['pair_strength'],
                'n_controls': len(controls),
            }
            for m in target_metrics:
                vals = [c[m] for c in controls]
                mu, sd, z, exc = compare(target_metrics[m], vals)
                row[f'{m}_target'] = target_metrics[m]
                row[f'{m}_control_mean'] = mu
                row[f'{m}_control_std'] = sd
                row[f'{m}_z'] = z
                row[f'{m}_exceed'] = exc
            full_rows.append(row)

d_full = pd.DataFrame(full_rows)
d_full.to_csv(OUT/'stepD_bridge_geometry_full.csv', index=False)

sum_rows = []
for crit in CRITERIA:
    sub = d_full[d_full['criterion'] == crit]
    for beta in BETAS:
        ss = sub[sub['beta'] == beta]
        row = {'criterion': crit, 'beta': beta}
        for m in ['neck_floor', 'neck_mid_ratio', 'axiality_ratio', 'continuity_frac', 'bridge_span_ratio', 'straightness']:
            row[f'{m}_exceed_mean'] = ss[f'{m}_exceed'].mean()
            row[f'{m}_z_mean'] = ss[f'{m}_z'].mean()
        sum_rows.append(row)
d_summary = pd.DataFrame(sum_rows)
d_summary.to_csv(OUT/'stepD_bridge_geometry_summary.csv', index=False)

# plot best candidates
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=False)
for crit in CRITERIA:
    sub = d_summary[d_summary['criterion'] == crit].sort_values('beta')
    axes[0].plot(sub['beta'], sub['axiality_ratio_exceed_mean'], marker='o', label=crit)
    axes[1].plot(sub['beta'], sub['continuity_frac_exceed_mean'], marker='o', label=crit)
    axes[2].plot(sub['beta'], sub['straightness_exceed_mean'], marker='o', label=crit)
for ax, title in zip(axes, ['Axiality', 'Continuity', 'Straightness']):
    ax.axhline(0.5, color='k', linestyle=':', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('beta')
axes[0].set_ylabel('mean exceedance vs same-shell controls')
axes[2].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT/'stepD_bridge_geometry_exceed.png', dpi=180)
plt.close(fig)

# Step E: add exactly one bridge metric to minimal kernel
# baseline features
dense = pd.read_csv(OUT/'dense_beta_branch_tracking_winners.csv')
dense = dense[dense['beta'].isin(BETAS)][['geometry','beta','criterion','boundary_ratio','q_family']].rename(columns={'boundary_ratio':'mean_boundary','q_family':'winner_family'})

closed = pd.read_csv(OUT/'closed_body_analogue_full.csv')[['geometry','beta','criterion','center_frac']]
opp = pd.read_csv(OUT/'opposite_center_locking_full.csv')
opp = opp[(opp['null_type']=='fft_phase') & (opp['shell']=='edge') & (opp['metric']=='direct_abs')][['geometry','beta','criterion','exceed']].rename(columns={'exceed':'opp_edge'})

base = dense.merge(closed, on=['geometry','beta','criterion']).merge(opp, on=['geometry','beta','criterion'])

# choose candidate bridge features from step D using z-scores
cand_cols = ['neck_floor_z', 'neck_mid_ratio_z', 'axiality_ratio_z', 'continuity_frac_z', 'bridge_span_ratio_z', 'straightness_z']
feat_df = d_full[['geometry','beta','criterion'] + cand_cols]
data = base.merge(feat_df, on=['geometry','beta','criterion'])
data['support_degree'] = data['winner_family'].map(support_degree)

def nearest_centroid_loocv(df, feat_cols, target='support_degree'):
    X = df[feat_cols].to_numpy(dtype=float)
    y = df[target].to_numpy()
    # z-score by train split only each time
    preds = []
    dists = []
    for i in range(len(df)):
        mask = np.ones(len(df), dtype=bool); mask[i] = False
        Xtr, ytr = X[mask], y[mask]
        Xte = X[~mask]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        Xtrn = (Xtr - mu) / sd
        Xten = (Xte - mu) / sd
        centroids = {cls: Xtrn[ytr == cls].mean(axis=0) for cls in sorted(set(ytr))}
        cds = {cls: float(np.linalg.norm(Xten[0] - c)) for cls, c in centroids.items()}
        pred = min(cds, key=cds.get)
        preds.append(pred)
        dists.append(cds[pred])
    preds = np.array(preds)
    acc = float(np.mean(preds == y))
    return acc, preds, np.array(dists)

baseline_feats = ['mean_boundary','center_frac','opp_edge']
base_acc, _, _ = nearest_centroid_loocv(data, baseline_feats)

rows = []
for c in cand_cols:
    acc, preds, dists = nearest_centroid_loocv(data, baseline_feats + [c])
    rows.append({
        'candidate_bridge_feature': c,
        'baseline_accuracy': base_acc,
        'augmented_accuracy': acc,
        'delta_accuracy': acc - base_acc,
        'mean_feature_abs': float(np.mean(np.abs(data[c]))),
        'corr_with_opp_edge': float(np.corrcoef(data[c], data['opp_edge'])[0,1]),
    })

e_summary = pd.DataFrame(rows).sort_values(['augmented_accuracy','delta_accuracy'], ascending=False)
e_summary.to_csv(OUT/'stepE_kernel_plus_bridge_summary.csv', index=False)

best_c = e_summary.iloc[0]['candidate_bridge_feature']
acc, preds, dists = nearest_centroid_loocv(data, baseline_feats + [best_c])
cases = data[['geometry','beta','criterion','winner_family','support_degree'] + baseline_feats + [best_c]].copy()
cases['pred_support_degree'] = preds
cases['correct'] = cases['pred_support_degree'] == cases['support_degree']
cases['distance'] = dists
cases.to_csv(OUT/'stepE_kernel_plus_bridge_cases.csv', index=False)

# plot
fig, ax = plt.subplots(figsize=(7.5,4.2))
tmp = e_summary.sort_values('augmented_accuracy', ascending=True)
ax.barh(tmp['candidate_bridge_feature'], tmp['augmented_accuracy'], color='tab:blue')
ax.axvline(base_acc, color='k', linestyle='--', linewidth=1, label=f'baseline={base_acc:.3f}')
ax.set_xlabel('LOOCV accuracy (support family)')
ax.set_title('Step E: baseline kernel + exactly one bridge metric')
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT/'stepE_kernel_plus_bridge_accuracy.png', dpi=180)
plt.close(fig)
