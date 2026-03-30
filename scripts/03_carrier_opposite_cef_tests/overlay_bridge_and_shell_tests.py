import argparse
import itertools
import math
import re
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy.ndimage import shift, convolve

# Reuse geometry and null-model helpers
spec_gpm = importlib.util.spec_from_file_location('gpm', '/mnt/data/geometry_phase_map.py')
gpm = importlib.util.module_from_spec(spec_gpm)
spec_gpm.loader.exec_module(gpm)

spec_pvn = importlib.util.spec_from_file_location('pvn', '/mnt/data/phase_vs_nullmodels.py')
pvn = importlib.util.module_from_spec(spec_pvn)
spec_pvn.loader.exec_module(pvn)

GEOMETRIES = pvn.GEOMETRIES
CRITERIA = pvn.CRITERIA
NULLS = pvn.NULLS
CENTER = (1, 1, 1)
RADII = {'face': 1.0, 'edge': math.sqrt(2.0), 'corner': math.sqrt(3.0)}


def shell_indices():
    corners, edges, faces = [], [], []
    for i, j, k in itertools.product(range(3), repeat=3):
        if (i, j, k) == CENTER:
            continue
        n_center = sum(v == 1 for v in (i, j, k))
        if n_center == 0:
            corners.append((i, j, k))
        elif n_center == 1:
            edges.append((i, j, k))
        else:
            faces.append((i, j, k))
    return {'corner': corners, 'edge': edges, 'face': faces}


SHELLS = shell_indices()


def opposite_pairs(indices):
    idx_set = set(indices)
    seen = set()
    pairs = []
    for a in indices:
        b = tuple((2 - np.array(a)).tolist())
        if b not in idx_set:
            continue
        key = tuple(sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b))
    return pairs


PAIRS = {k: opposite_pairs(v) for k, v in SHELLS.items()}
NONOP_PAIRS = {}
for shell, inds in SHELLS.items():
    all_pairs = [tuple(sorted(p)) for p in itertools.combinations(inds, 2)]
    opp = {tuple(sorted(p)) for p in PAIRS[shell]}
    NONOP_PAIRS[shell] = [p for p in all_pairs if p not in opp]


# Dense xyz carrier kernel on the full winner field
KERNEL = np.zeros((3, 3, 3), dtype=float)
for a in (-1, 1):
    for b in (-1, 1):
        for c in (-1, 1):
            KERNEL[a + 1, b + 1, c + 1] = (a * b * c) / 8.0


def dense_xyz_carrier(u: np.ndarray) -> np.ndarray:
    G = convolve(u, KERNEL, mode='constant', cval=0.0)
    return G[1:-1, 1:-1, 1:-1]


def q_family(label: str) -> str:
    if label == 'const':
        return 'const'
    vals = []
    for part in label.split('+'):
        m = re.match(r'([XYZ])(\d+)', part)
        if m is None:
            raise ValueError(f'Bad q label: {label!r}')
        vals.append(int(m.group(2)))
    vals = tuple(sorted(vals))
    return f'{len(vals)}-axis:{vals}'


def local_metrics_from_xyz(xyz: np.ndarray):
    fft = np.fft.fftn(xyz)
    power = np.abs(fft) ** 2
    power[0, 0, 0] = 0.0
    q = np.unravel_index(np.argmax(power), power.shape)
    q_label = gpm.q_label(q)
    q_contrast = float(power[q] / (power.sum() + 1e-12))
    mean_abs_xyz = float(np.mean(np.abs(xyz)))
    return q_label, q_family(q_label), q_contrast, mean_abs_xyz


def solve_mode_full(lengths, beta, ncell=3, pts_per_cell=5, modes=10):
    Lx, Ly, Lz = lengths
    Nx = ncell * pts_per_cell + 1
    Ny = ncell * pts_per_cell + 1
    Nz = ncell * pts_per_cell + 1
    A, hx, hy, hz = gpm.laplacian_3d(Nx, Ny, Nz, ncell * Lx, ncell * Ly, ncell * Lz, beta)
    vals, vecs = spla.eigsh(A, k=modes, which='SM', tol=1e-6)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    rows = []
    for idx in range(modes):
        u = vecs[:, idx].reshape((Nx, Ny, Nz))
        xyz = gpm.local_xyz_array(u, ncell=ncell, pts_per_cell=pts_per_cell)
        anis = gpm.density_anisotropy(u, hx, hy, hz)
        B = gpm.boundary_ratio(u, hx, hy, hz)
        q_label, qfam, q_contrast, mean_abs_xyz = local_metrics_from_xyz(xyz)
        rows.append({
            'local_mode_index': idx + 1,
            'eigenvalue': float(vals[idx]),
            'u': u,
            'xyz': xyz,
            'field_anisotropy': anis,
            'boundary_ratio': B,
            'dominant_q': q_label,
            'q_family': qfam,
            'q_contrast': q_contrast,
            'mean_abs_xyz': mean_abs_xyz,
            'score_q': mean_abs_xyz * q_contrast,
            'score_iso': mean_abs_xyz * anis,
            'score_q_only': q_contrast,
            'score_xyz_only': mean_abs_xyz,
        })
    return rows


def pick_winners(rows):
    sub = [r for r in rows if r['local_mode_index'] > 1]
    out = {}
    for crit, col in CRITERIA.items():
        out[crit] = max(sub, key=lambda r: r[col])
    return out


def pair_strength(xyz, pair):
    a, b = pair
    va = abs(float(xyz[a]))
    vb = abs(float(xyz[b]))
    coh = 2.0 * va * vb / (va * va + vb * vb + 1e-12)
    return float(math.sqrt(va * vb) * coh)


def best_opposite_pair(xyz):
    best = None
    for shell, pairs in PAIRS.items():
        for pair in pairs:
            score = pair_strength(xyz, pair)
            if best is None or score > best['pair_strength']:
                best = {
                    'target_shell': shell,
                    'target_pair': pair,
                    'pair_strength': score,
                }
    return best


def cell_center(idx, pts_per_cell=5):
    i, j, k = idx
    return np.array([i * pts_per_cell + pts_per_cell // 2,
                     j * pts_per_cell + pts_per_cell // 2,
                     k * pts_per_cell + pts_per_cell // 2], dtype=float)


def overlay_carrier_score(u, pair, pts_per_cell=5, target_sep=8, pad=10):
    # Compose in carrier space, not on raw density only
    G0 = np.abs(dense_xyz_carrier(u))
    base = np.zeros(tuple(np.array(G0.shape) + 2 * pad), dtype=float)
    base[pad:pad + G0.shape[0], pad:pad + G0.shape[1], pad:pad + G0.shape[2]] = G0
    off = np.array([pad, pad, pad], dtype=float)

    p1, p2 = [(cell_center(idx, pts_per_cell) - 1) + off for idx in pair]
    center = np.array(base.shape) // 2
    t1 = center + np.array([-target_sep // 2, 0, 0], dtype=float)
    t2 = center + np.array([ target_sep // 2, 0, 0], dtype=float)
    s1 = t1 - p1
    s2 = t2 - p2

    O = shift(base, s1, order=1, mode='constant', cval=0.0, prefilter=False)
    O += shift(base, s2, order=1, mode='constant', cval=0.0, prefilter=False)

    c = center.astype(int)
    t1 = t1.astype(int)
    t2 = t2.astype(int)
    mid = float(O[tuple(c)])
    lobe = float((O[tuple(t1)] + O[tuple(t2)]) / 2.0)
    xs = np.arange(min(t1[0], t2[0]), max(t1[0], t2[0]) + 1)
    line = O[xs, c[1], c[2]]
    off_line = 0.25 * (
        O[xs, c[1] + 2, c[2]] + O[xs, c[1] - 2, c[2]] +
        O[xs, c[1], c[2] + 2] + O[xs, c[1], c[2] - 2]
    )
    bridge_line_ratio = float(np.mean(line) / (np.mean(off_line) + 1e-12))
    midpoint_ratio = float(mid / (lobe + 1e-12))
    return {
        'bridge_line_ratio': bridge_line_ratio,
        'midpoint_ratio': midpoint_ratio,
        'mid_carrier': mid,
        'lobe_carrier': lobe,
    }


def shell_mass_metrics(xyz):
    masses = {
        shell: float(np.sum(np.abs([xyz[idx] for idx in inds])))
        for shell, inds in SHELLS.items()
    }
    total = sum(masses.values()) + 1e-12
    dominant_shell = max(masses, key=masses.get)
    shell_purity = masses[dominant_shell] / total
    rmean = sum(masses[s] * RADII[s] for s in masses) / total
    rvar = sum(masses[s] * (RADII[s] - rmean) ** 2 for s in masses) / total
    return {
        'dominant_shell': dominant_shell,
        'shell_purity': float(shell_purity),
        'radius_mean': float(rmean),
        'radius_std': float(np.sqrt(rvar)),
        'face_mass_frac': masses['face'] / total,
        'edge_mass_frac': masses['edge'] / total,
        'corner_mass_frac': masses['corner'] / total,
    }


def compare_against_surrogates(real_val, sur_vals, lower_is_better=False):
    sur_vals = np.asarray(sur_vals, dtype=float)
    mu = float(np.mean(sur_vals))
    sd = float(np.std(sur_vals, ddof=0))
    if lower_is_better:
        z = float((mu - real_val) / (sd + 1e-12))
        exceed = float(np.mean(real_val < sur_vals))
    else:
        z = float((real_val - mu) / (sd + 1e-12))
        exceed = float(np.mean(real_val > sur_vals))
    return mu, sd, z, exceed


def plot_overlay(summary, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    for crit in ['score_q', 'score_iso', 'q_only', 'xyz_only']:
        sub = summary[summary['criterion'] == crit].sort_values('beta')
        axes[0].plot(sub['beta'], sub['bridge_exceed_mean'], marker='o', label=crit)
        axes[1].plot(sub['beta'], sub['midpoint_exceed_mean'], marker='o', label=crit)
    axes[0].axhline(0.5, color='k', linestyle=':', linewidth=1)
    axes[1].axhline(0.5, color='k', linestyle=':', linewidth=1)
    axes[0].set_title('Overlay bridge: target pair exceeds matched controls')
    axes[1].set_title('Overlay midpoint: target pair exceeds matched controls')
    for ax in axes:
        ax.set_xlabel('beta')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('exceedance fraction')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.07), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_shell(summary, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)
    sfft = summary[summary['null_type'] == 'fft_phase']
    for crit in ['score_q', 'score_iso', 'q_only', 'xyz_only']:
        sub = sfft[sfft['criterion'] == crit].sort_values('beta')
        axes[0].plot(sub['beta'], sub['shell_purity_exceed_mean'], marker='o', label=crit)
        axes[1].plot(sub['beta'], sub['radius_std_exceed_mean'], marker='o', label=crit)
    axes[0].axhline(0.5, color='k', linestyle=':', linewidth=1)
    axes[1].axhline(0.5, color='k', linestyle=':', linewidth=1)
    axes[0].set_title('Carrier-shell purity vs fft-phase surrogate')
    axes[1].set_title('Carrier-shell tightness vs fft-phase surrogate')
    for ax in axes:
        ax.set_xlabel('beta')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('exceedance fraction')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.07), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_shell_dominance(real_shell_df, out_png):
    # dominant shell counts across geometries for each beta and criterion
    criteria = ['score_q', 'score_iso', 'q_only', 'xyz_only']
    betas = sorted(real_shell_df['beta'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    axes = axes.ravel()
    colors = {'corner': '#aa3377', 'edge': '#4477aa', 'face': '#44aa66'}
    for ax, crit in zip(axes, criteria):
        sub = real_shell_df[real_shell_df['criterion'] == crit]
        bottom = np.zeros(len(betas))
        for shell in ['corner', 'edge', 'face']:
            vals = [np.mean(sub[sub['beta'] == b]['dominant_shell'] == shell) for b in betas]
            ax.bar(betas, vals, bottom=bottom, color=colors[shell], width=0.32, label=shell)
            bottom += np.array(vals)
        ax.set_title(crit)
        ax.grid(True, axis='y', alpha=0.3)
    axes[0].set_ylabel('fraction of geometries')
    axes[2].set_ylabel('fraction of geometries')
    axes[2].set_xlabel('beta')
    axes[3].set_xlabel('beta')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.02))
    fig.suptitle('Dominant carrier shell across geometries')
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--betas', type=str, default='0,1,1.5,2,5,10')
    parser.add_argument('--pts-per-cell', type=int, default=5)
    parser.add_argument('--modes', type=int, default=10)
    parser.add_argument('--shell-draws', type=int, default=48)
    parser.add_argument('--max-controls', type=int, default=24)
    parser.add_argument('--seed', type=int, default=20260320)
    parser.add_argument('--overlay-full-csv', type=str, default='/mnt/data/overlay_bridge_full.csv')
    parser.add_argument('--overlay-summary-csv', type=str, default='/mnt/data/overlay_bridge_summary.csv')
    parser.add_argument('--overlay-plot', type=str, default='/mnt/data/overlay_bridge_exceed.png')
    parser.add_argument('--shell-full-csv', type=str, default='/mnt/data/carrier_shell_stability_full.csv')
    parser.add_argument('--shell-summary-csv', type=str, default='/mnt/data/carrier_shell_stability_summary.csv')
    parser.add_argument('--shell-plot', type=str, default='/mnt/data/carrier_shell_stability_exceed.png')
    parser.add_argument('--shell-dominance-plot', type=str, default='/mnt/data/carrier_shell_dominance.png')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    betas = [float(x) for x in args.betas.split(',') if x.strip()]

    overlay_rows = []
    shell_draw_rows = []
    real_shell_rows = []

    for geom, lengths in GEOMETRIES:
        for beta in betas:
            modes = solve_mode_full(lengths, beta, pts_per_cell=args.pts_per_cell, modes=args.modes)
            winners = pick_winners(modes)
            for criterion, winner in winners.items():
                xyz = winner['xyz']
                u = winner['u']

                # --- Test 1: overlay bridge on strongest opposite carrier pair ---
                target = best_opposite_pair(xyz)
                control_pairs = NONOP_PAIRS[target['target_shell']]
                if len(control_pairs) > args.max_controls:
                    sel = rng.choice(len(control_pairs), size=args.max_controls, replace=False)
                    control_pairs = [control_pairs[int(i)] for i in sel]
                target_scores = overlay_carrier_score(u, target['target_pair'], pts_per_cell=args.pts_per_cell)
                control_scores = [overlay_carrier_score(u, pair, pts_per_cell=args.pts_per_cell) for pair in control_pairs]
                cdf = pd.DataFrame(control_scores)
                overlay_rows.append({
                    'geometry': geom,
                    'beta': beta,
                    'criterion': criterion,
                    'target_shell': target['target_shell'],
                    'target_pair': str(target['target_pair']),
                    'pair_strength': target['pair_strength'],
                    'bridge_target': target_scores['bridge_line_ratio'],
                    'bridge_control_mean': float(cdf['bridge_line_ratio'].mean()),
                    'bridge_control_std': float(cdf['bridge_line_ratio'].std(ddof=0)),
                    'bridge_exceed': float(np.mean(target_scores['bridge_line_ratio'] > cdf['bridge_line_ratio'])),
                    'bridge_z': float((target_scores['bridge_line_ratio'] - cdf['bridge_line_ratio'].mean()) / (cdf['bridge_line_ratio'].std(ddof=0) + 1e-12)),
                    'midpoint_target': target_scores['midpoint_ratio'],
                    'midpoint_control_mean': float(cdf['midpoint_ratio'].mean()),
                    'midpoint_control_std': float(cdf['midpoint_ratio'].std(ddof=0)),
                    'midpoint_exceed': float(np.mean(target_scores['midpoint_ratio'] > cdf['midpoint_ratio'])),
                    'midpoint_z': float((target_scores['midpoint_ratio'] - cdf['midpoint_ratio'].mean()) / (cdf['midpoint_ratio'].std(ddof=0) + 1e-12)),
                    'n_controls': len(cdf),
                    'winner_q': winner['dominant_q'],
                    'winner_family': winner['q_family'],
                })

                # --- Test 2: carrier-shell stability ---
                real_shell = shell_mass_metrics(xyz)
                real_shell_rows.append({
                    'geometry': geom,
                    'beta': beta,
                    'criterion': criterion,
                    **real_shell,
                    'winner_q': winner['dominant_q'],
                    'winner_family': winner['q_family'],
                })
                for null_name, fn in NULLS.items():
                    sur_purity = []
                    sur_rstd = []
                    sur_dom = []
                    for draw in range(args.shell_draws):
                        sxyz = fn(xyz, rng)
                        sm = shell_mass_metrics(sxyz)
                        shell_draw_rows.append({
                            'geometry': geom,
                            'beta': beta,
                            'criterion': criterion,
                            'null_type': null_name,
                            'draw': draw,
                            'real_dominant_shell': real_shell['dominant_shell'],
                            'real_shell_purity': real_shell['shell_purity'],
                            'real_radius_std': real_shell['radius_std'],
                            'sur_dominant_shell': sm['dominant_shell'],
                            'sur_shell_purity': sm['shell_purity'],
                            'sur_radius_std': sm['radius_std'],
                        })
                        sur_purity.append(sm['shell_purity'])
                        sur_rstd.append(sm['radius_std'])
                        sur_dom.append(sm['dominant_shell'])

    overlay_df = pd.DataFrame(overlay_rows)
    overlay_summary = overlay_df.groupby(['criterion', 'beta']).agg(
        bridge_exceed_mean=('bridge_exceed', 'mean'),
        bridge_z_mean=('bridge_z', 'mean'),
        bridge_target_mean=('bridge_target', 'mean'),
        bridge_control_mean=('bridge_control_mean', 'mean'),
        midpoint_exceed_mean=('midpoint_exceed', 'mean'),
        midpoint_z_mean=('midpoint_z', 'mean'),
        midpoint_target_mean=('midpoint_target', 'mean'),
        midpoint_control_mean=('midpoint_control_mean', 'mean'),
    ).reset_index()
    overlay_df.to_csv(args.overlay_full_csv, index=False)
    overlay_summary.to_csv(args.overlay_summary_csv, index=False)

    shell_full = pd.DataFrame(shell_draw_rows)
    real_shell_df = pd.DataFrame(real_shell_rows)
    shell_summary_rows = []
    for (geom, beta, criterion, null_name), grp in shell_full.groupby(['geometry', 'beta', 'criterion', 'null_type']):
        real_purity = float(grp['real_shell_purity'].iloc[0])
        real_rstd = float(grp['real_radius_std'].iloc[0])
        real_dom = str(grp['real_dominant_shell'].iloc[0])
        purity_mu, purity_sd, purity_z, purity_exceed = compare_against_surrogates(real_purity, grp['sur_shell_purity'])
        rstd_mu, rstd_sd, rstd_z, rstd_exceed = compare_against_surrogates(real_rstd, grp['sur_radius_std'], lower_is_better=True)
        dom_match = float(np.mean(grp['sur_dominant_shell'] == real_dom))
        shell_summary_rows.append({
            'geometry': geom,
            'beta': beta,
            'criterion': criterion,
            'null_type': null_name,
            'real_dominant_shell': real_dom,
            'real_shell_purity': real_purity,
            'real_radius_std': real_rstd,
            'sur_shell_purity_mean': purity_mu,
            'sur_shell_purity_std': purity_sd,
            'shell_purity_z': purity_z,
            'shell_purity_exceed': purity_exceed,
            'sur_radius_std_mean': rstd_mu,
            'sur_radius_std_std': rstd_sd,
            'radius_std_z': rstd_z,
            'radius_std_exceed': rstd_exceed,
            'dominant_shell_match': dom_match,
        })
    shell_summary = pd.DataFrame(shell_summary_rows)
    shell_agg = shell_summary.groupby(['criterion', 'beta', 'null_type']).agg(
        shell_purity_exceed_mean=('shell_purity_exceed', 'mean'),
        shell_purity_z_mean=('shell_purity_z', 'mean'),
        radius_std_exceed_mean=('radius_std_exceed', 'mean'),
        radius_std_z_mean=('radius_std_z', 'mean'),
        dominant_shell_match_mean=('dominant_shell_match', 'mean'),
        real_shell_purity_mean=('real_shell_purity', 'mean'),
        real_radius_std_mean=('real_radius_std', 'mean'),
        sur_shell_purity_mean=('sur_shell_purity_mean', 'mean'),
        sur_radius_std_mean=('sur_radius_std_mean', 'mean'),
    ).reset_index()
    shell_full.to_csv(args.shell_full_csv, index=False)
    shell_agg.to_csv(args.shell_summary_csv, index=False)

    plot_overlay(overlay_summary, args.overlay_plot)
    plot_shell(shell_agg, args.shell_plot)
    plot_shell_dominance(real_shell_df, args.shell_dominance_plot)

    print('Saved:')
    print(args.overlay_full_csv)
    print(args.overlay_summary_csv)
    print(args.shell_full_csv)
    print(args.shell_summary_csv)
    print('\nOverlay summary:')
    print(overlay_summary.to_string(index=False))
    print('\nCarrier-shell summary (aggregated):')
    print(shell_agg.to_string(index=False))


if __name__ == '__main__':
    main()
