import argparse
import math
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

# Load helpers from previous scripts

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

gpm = load_module('gpm', '/mnt/data/geometry_phase_map.py')
pvn = load_module('pvn', '/mnt/data/phase_vs_nullmodels.py')
ov = load_module('ov', '/mnt/data/overlay_bridge_and_shell_tests.py')

GEOMETRIES = pvn.GEOMETRIES
CRITERIA = pvn.CRITERIA
NULLS = pvn.NULLS
SHELLS = ov.SHELLS


def solve_mode_full(lengths, beta, ncell=3, pts_per_cell=5, modes=10):
    return ov.solve_mode_full(lengths, beta, ncell=ncell, pts_per_cell=pts_per_cell, modes=modes)


def pick_winners(rows):
    return ov.pick_winners(rows)


def pair_strength(xyz, pair):
    return ov.pair_strength(xyz, pair)


def overlay_carrier_score(u, pair, pts_per_cell=5):
    return ov.overlay_carrier_score(u, pair, pts_per_cell=pts_per_cell)


def dense_xyz_carrier(u):
    return ov.dense_xyz_carrier(u)


def shell_mass_metrics(xyz):
    return ov.shell_mass_metrics(xyz)


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


def top_points_nms(G, topk=24, min_dist=2):
    A = np.abs(G).copy()
    pts = []
    vals = []
    shp = np.array(A.shape)
    for _ in range(topk):
        idx = np.unravel_index(np.argmax(A), A.shape)
        v = float(A[idx])
        if v <= 0.0:
            break
        pts.append(np.array(idx, dtype=float))
        vals.append(v)
        lo = np.maximum(np.array(idx) - min_dist, 0)
        hi = np.minimum(np.array(idx) + min_dist + 1, shp)
        A[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 0.0
    return np.array(pts, dtype=float), np.array(vals, dtype=float)


def orientation_tensor(coords, vals, shape):
    center = (np.array(shape, dtype=float) - 1.0) / 2.0
    M = np.zeros((3, 3), dtype=float)
    wsum = 0.0
    for c, v in zip(coords, vals):
        d = np.array(c, dtype=float) - center
        r = float(np.linalg.norm(d))
        if r < 1e-12:
            continue
        u = d / r
        M += float(v) * np.outer(u, u)
        wsum += float(v)
    if wsum < 1e-12:
        return np.eye(3) / 3.0
    return M / wsum


def tensor_align(A, B):
    return float(np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12))


def shell_of(idx):
    for shell, inds in SHELLS.items():
        if idx in inds:
            return shell
    raise KeyError(idx)


def same_shell_angle_control(winner, pts_per_cell=5):
    """Fix the stronger endpoint of the strongest opposite pair and compare only
    against other points on the *same shell*. This isolates angle much more than
    the earlier pairwise test."""
    xyz = winner['xyz']
    u = winner['u']
    target = ov.best_opposite_pair(xyz)
    p, q = target['target_pair']
    if abs(float(xyz[q])) > abs(float(xyz[p])):
        p, q = q, p
    shell = target['target_shell']
    controls = [c for c in SHELLS[shell] if c not in [p, q]]
    target_pair_score = pair_strength(xyz, (p, q))
    control_scores = np.array([pair_strength(xyz, (p, c)) for c in controls], dtype=float)
    target_bridge = overlay_carrier_score(u, tuple(sorted((p, q))), pts_per_cell=pts_per_cell)['bridge_line_ratio']
    control_bridges = np.array([
        overlay_carrier_score(u, tuple(sorted((p, c))), pts_per_cell=pts_per_cell)['bridge_line_ratio']
        for c in controls
    ], dtype=float)
    return {
        'anchor': str(p),
        'opposite': str(q),
        'shell': shell,
        'target_pair_strength': float(target_pair_score),
        'control_pair_strength_mean': float(control_scores.mean()),
        'pair_exceed': float(np.mean(target_pair_score > control_scores)),
        'pair_z': float((target_pair_score - control_scores.mean()) / (control_scores.std(ddof=0) + 1e-12)),
        'target_bridge': float(target_bridge),
        'control_bridge_mean': float(control_bridges.mean()),
        'bridge_exceed': float(np.mean(target_bridge > control_bridges)),
        'bridge_z': float((target_bridge - control_bridges.mean()) / (control_bridges.std(ddof=0) + 1e-12)),
        'n_controls': len(controls),
    }


def cross_layer_alignment_metrics(u):
    G = dense_xyz_carrier(u)
    coords, vals = top_points_nms(G, topk=24, min_dist=2)
    if len(vals) < 3:
        return {
            'layer_align_mean': np.nan,
            'layer_align_min': np.nan,
            'layer_aniso_mean': np.nan,
            'nmax': int(len(vals)),
        }
    bins = np.array_split(np.arange(len(vals)), 3)
    tensors = [orientation_tensor(coords[b], vals[b], G.shape) for b in bins]
    aligns = [tensor_align(tensors[0], tensors[1]), tensor_align(tensors[0], tensors[2]), tensor_align(tensors[1], tensors[2])]
    anisos = []
    for T in tensors:
        ev = np.linalg.eigvalsh(T)
        ev = np.clip(ev, 1e-14, None)
        anisos.append(float(ev.min() / ev.max()))
    return {
        'layer_align_mean': float(np.mean(aligns)),
        'layer_align_min': float(np.min(aligns)),
        'layer_aniso_mean': float(np.mean(anisos)),
        'nmax': int(len(vals)),
    }


def closed_body_metrics(winner):
    G = np.abs(dense_xyz_carrier(winner['u']))
    coords, vals = top_points_nms(G, topk=24, min_dist=2)
    shape = np.array(G.shape, dtype=float)
    center = (shape - 1.0) / 2.0
    grid = np.indices(G.shape).reshape(3, -1).T.astype(float)
    d = np.linalg.norm(grid - center, axis=1)
    R = float(d.max())
    total = float(G.sum()) + 1e-12
    center_frac = float(G.reshape(-1)[d <= 0.30 * R].sum() / total)
    peak_contrast = float(vals[0] / (G.mean() + 1e-12)) if len(vals) else 0.0
    top_frac = float(vals[:min(8, len(vals))].sum() / total) if len(vals) else 0.0
    T = orientation_tensor(coords, vals, G.shape)
    ev = np.linalg.eigvalsh(T)
    ev = np.clip(ev, 1e-14, None)
    hotspot_iso = float(ev.min() / ev.max())
    sm = shell_mass_metrics(winner['xyz'])
    shell_diffuse = float(1.0 - sm['shell_purity'])
    closed_score = float(center_frac * hotspot_iso * shell_diffuse / (1.0 + 0.1 * peak_contrast))
    return {
        'center_frac': center_frac,
        'peak_contrast': peak_contrast,
        'top_frac': top_frac,
        'hotspot_iso': hotspot_iso,
        'shell_diffuse': shell_diffuse,
        'closed_score': closed_score,
        'nmax': int(len(vals)),
    }


def plot_same_shell(summary, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for crit in ['score_q', 'score_iso', 'q_only', 'xyz_only']:
        sub = summary[summary['criterion'] == crit].sort_values('beta')
        axes[0].plot(sub['beta'], sub['pair_exceed_mean'], marker='o', label=crit)
        axes[1].plot(sub['beta'], sub['bridge_exceed_mean'], marker='o', label=crit)
    for ax in axes:
        ax.axhline(0.5, color='k', linestyle=':', linewidth=1)
        ax.set_xlabel('beta')
        ax.grid(True, alpha=0.3)
    axes[0].set_title('Same-shell angle control: pair locking')
    axes[1].set_title('Same-shell angle control: bridge score')
    axes[0].set_ylabel('exceedance fraction')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.06), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_cross_layer(summary, out_png):
    sfft = summary[summary['null_type'] == 'fft_phase']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for crit in ['score_q', 'score_iso', 'q_only', 'xyz_only']:
        sub = sfft[sfft['criterion'] == crit].sort_values('beta')
        axes[0].plot(sub['beta'], sub['align_exceed_mean'], marker='o', label=crit)
        axes[1].plot(sub['beta'], sub['align_z_mean'], marker='o', label=crit)
    axes[0].axhline(0.5, color='k', linestyle=':', linewidth=1)
    axes[1].axhline(0.0, color='k', linestyle=':', linewidth=1)
    axes[0].set_title('Cross-layer carrier alignment vs fft-phase')
    axes[1].set_title('Cross-layer carrier alignment z-score')
    for ax in axes:
        ax.set_xlabel('beta')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('exceedance fraction')
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.06), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)


def plot_closed(summary, out_png):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    metrics = [
        ('closed_score_mean', 'closed score'),
        ('center_frac_mean', 'center mass fraction'),
        ('shell_diffuse_mean', 'shell diffuseness'),
        ('hotspot_iso_mean', 'hotspot isotropy'),
        ('peak_contrast_mean', 'peak contrast'),
        ('top_frac_mean', 'top-8 mass fraction'),
    ]
    axes = axes.ravel()
    for ax, (col, title) in zip(axes, metrics):
        for crit in ['score_q', 'score_iso', 'q_only', 'xyz_only']:
            sub = summary[summary['criterion'] == crit].sort_values('beta')
            ax.plot(sub['beta'], sub[col], marker='o', label=crit)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('beta')
    axes[0].set_ylabel('mean value')
    axes[3].set_ylabel('mean value')
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.02), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--betas', type=str, default='0,1,1.5,2,5,10')
    parser.add_argument('--pts-per-cell', type=int, default=5)
    parser.add_argument('--modes', type=int, default=10)
    parser.add_argument('--null-draws', type=int, default=24)
    parser.add_argument('--seed', type=int, default=20260321)
    parser.add_argument('--same-shell-full-csv', type=str, default='/mnt/data/same_shell_angle_control_full.csv')
    parser.add_argument('--same-shell-summary-csv', type=str, default='/mnt/data/same_shell_angle_control_summary.csv')
    parser.add_argument('--same-shell-plot', type=str, default='/mnt/data/same_shell_angle_control.png')
    parser.add_argument('--cross-layer-full-csv', type=str, default='/mnt/data/cross_layer_alignment_full.csv')
    parser.add_argument('--cross-layer-summary-csv', type=str, default='/mnt/data/cross_layer_alignment_summary.csv')
    parser.add_argument('--cross-layer-plot', type=str, default='/mnt/data/cross_layer_alignment_fft.png')
    parser.add_argument('--closed-full-csv', type=str, default='/mnt/data/closed_body_analogue_full.csv')
    parser.add_argument('--closed-summary-csv', type=str, default='/mnt/data/closed_body_analogue_summary.csv')
    parser.add_argument('--closed-plot', type=str, default='/mnt/data/closed_body_analogue.png')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    betas = [float(x) for x in args.betas.split(',') if x.strip()]

    same_shell_rows = []
    cross_layer_draw_rows = []
    closed_rows = []

    for geom, lengths in GEOMETRIES:
        for beta in betas:
            rows = solve_mode_full(lengths, beta, pts_per_cell=args.pts_per_cell, modes=args.modes)
            winners = pick_winners(rows)
            for criterion, winner in winners.items():
                # Test 3a: same-shell different-angle control
                ss = same_shell_angle_control(winner, pts_per_cell=args.pts_per_cell)
                same_shell_rows.append({
                    'geometry': geom,
                    'beta': beta,
                    'criterion': criterion,
                    **ss,
                    'winner_q': winner['dominant_q'],
                    'winner_family': winner['q_family'],
                })

                # Test 4: cross-layer carrier alignment vs nulls
                real_align = cross_layer_alignment_metrics(winner['u'])
                for null_name, fn in NULLS.items():
                    vals_align = []
                    vals_min = []
                    vals_aniso = []
                    for draw in range(args.null_draws):
                        sxyz = fn(winner['xyz'], rng)
                        # build surrogate 3x3x3 carrier back into a minimal field via xyz only is not possible;
                        # so keep the current model honest: alignment nulls are applied on the dense carrier field
                        # implied by the xyz readout by replacing the winner xyz before evaluating local shell/orientation.
                        # We construct a surrogate dense carrier proxy directly from the shuffled xyz cube.
                        # Use the same NMS/tensor logic on the 3x3x3 surrogate itself.
                        # To stay comparable, embed surrogate xyz in a 3x3x3 array and use it as carrier field.
                        coords, vals = top_points_nms(np.abs(sxyz), topk=24, min_dist=1)
                        if len(vals) < 3:
                            align_mean = np.nan
                            align_min = np.nan
                            align_aniso = np.nan
                        else:
                            bins = np.array_split(np.arange(len(vals)), 3)
                            tensors = [orientation_tensor(coords[b], vals[b], sxyz.shape) for b in bins]
                            aligns = [tensor_align(tensors[0], tensors[1]), tensor_align(tensors[0], tensors[2]), tensor_align(tensors[1], tensors[2])]
                            align_mean = float(np.mean(aligns))
                            align_min = float(np.min(aligns))
                            aa = []
                            for T in tensors:
                                ev = np.linalg.eigvalsh(T)
                                ev = np.clip(ev, 1e-14, None)
                                aa.append(float(ev.min() / ev.max()))
                            align_aniso = float(np.mean(aa))
                        vals_align.append(align_mean)
                        vals_min.append(align_min)
                        vals_aniso.append(align_aniso)
                    mu, sd, z, exceed = compare_against_surrogates(real_align['layer_align_mean'], vals_align)
                    mu_min, sd_min, z_min, exceed_min = compare_against_surrogates(real_align['layer_align_min'], vals_min)
                    mu_an, sd_an, z_an, exceed_an = compare_against_surrogates(real_align['layer_aniso_mean'], vals_aniso)
                    cross_layer_draw_rows.append({
                        'geometry': geom,
                        'beta': beta,
                        'criterion': criterion,
                        'null_type': null_name,
                        'real_align_mean': real_align['layer_align_mean'],
                        'real_align_min': real_align['layer_align_min'],
                        'real_align_aniso': real_align['layer_aniso_mean'],
                        'sur_align_mean': mu,
                        'sur_align_std': sd,
                        'align_z': z,
                        'align_exceed': exceed,
                        'sur_align_min_mean': mu_min,
                        'sur_align_min_std': sd_min,
                        'align_min_z': z_min,
                        'align_min_exceed': exceed_min,
                        'sur_align_aniso_mean': mu_an,
                        'sur_align_aniso_std': sd_an,
                        'align_aniso_z': z_an,
                        'align_aniso_exceed': exceed_an,
                        'nmax_real': real_align['nmax'],
                    })

                # Test 5: closed-body analogue baseline (closest in-model analogue to old exponent-2 intuition)
                cb = closed_body_metrics(winner)
                closed_rows.append({
                    'geometry': geom,
                    'beta': beta,
                    'criterion': criterion,
                    **cb,
                    'winner_q': winner['dominant_q'],
                    'winner_family': winner['q_family'],
                })

    same_shell_df = pd.DataFrame(same_shell_rows)
    same_shell_summary = same_shell_df.groupby(['criterion', 'beta']).agg(
        pair_exceed_mean=('pair_exceed', 'mean'),
        pair_z_mean=('pair_z', 'mean'),
        bridge_exceed_mean=('bridge_exceed', 'mean'),
        bridge_z_mean=('bridge_z', 'mean'),
        target_pair_strength_mean=('target_pair_strength', 'mean'),
        control_pair_strength_mean=('control_pair_strength_mean', 'mean'),
        target_bridge_mean=('target_bridge', 'mean'),
        control_bridge_mean=('control_bridge_mean', 'mean'),
    ).reset_index()

    cross_layer_df = pd.DataFrame(cross_layer_draw_rows)
    cross_layer_summary = cross_layer_df.groupby(['criterion', 'beta', 'null_type']).agg(
        align_exceed_mean=('align_exceed', 'mean'),
        align_z_mean=('align_z', 'mean'),
        align_min_exceed_mean=('align_min_exceed', 'mean'),
        align_min_z_mean=('align_min_z', 'mean'),
        align_aniso_exceed_mean=('align_aniso_exceed', 'mean'),
        align_aniso_z_mean=('align_aniso_z', 'mean'),
        real_align_mean=('real_align_mean', 'mean'),
        sur_align_mean=('sur_align_mean', 'mean'),
        real_align_aniso=('real_align_aniso', 'mean'),
        sur_align_aniso_mean=('sur_align_aniso_mean', 'mean'),
    ).reset_index()

    closed_df = pd.DataFrame(closed_rows)
    closed_summary = closed_df.groupby(['criterion', 'beta']).agg(
        closed_score_mean=('closed_score', 'mean'),
        center_frac_mean=('center_frac', 'mean'),
        peak_contrast_mean=('peak_contrast', 'mean'),
        top_frac_mean=('top_frac', 'mean'),
        hotspot_iso_mean=('hotspot_iso', 'mean'),
        shell_diffuse_mean=('shell_diffuse', 'mean'),
        nmax_mean=('nmax', 'mean'),
    ).reset_index()

    same_shell_df.to_csv(args.same_shell_full_csv, index=False)
    same_shell_summary.to_csv(args.same_shell_summary_csv, index=False)
    cross_layer_df.to_csv(args.cross_layer_full_csv, index=False)
    cross_layer_summary.to_csv(args.cross_layer_summary_csv, index=False)
    closed_df.to_csv(args.closed_full_csv, index=False)
    closed_summary.to_csv(args.closed_summary_csv, index=False)

    plot_same_shell(same_shell_summary, args.same_shell_plot)
    plot_cross_layer(cross_layer_summary, args.cross_layer_plot)
    plot_closed(closed_summary, args.closed_plot)

    print('Saved:')
    for p in [
        args.same_shell_full_csv, args.same_shell_summary_csv, args.same_shell_plot,
        args.cross_layer_full_csv, args.cross_layer_summary_csv, args.cross_layer_plot,
        args.closed_full_csv, args.closed_summary_csv, args.closed_plot,
    ]:
        print(p)
    print('\nSame-shell summary:')
    print(same_shell_summary.to_string(index=False))
    print('\nCross-layer summary:')
    print(cross_layer_summary.to_string(index=False))
    print('\nClosed-body analogue summary:')
    print(closed_summary.to_string(index=False))

if __name__ == '__main__':
    main()
