import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util

spec = importlib.util.spec_from_file_location('pvn','/mnt/data/phase_vs_nullmodels.py')
pvn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pvn)

GEOMETRIES = pvn.GEOMETRIES
CRITERIA = pvn.CRITERIA
NULLS = {
    'cell_shuffle': pvn.surrogate_cell_shuffle,
    'shell_shuffle': pvn.surrogate_shell_shuffle,
    'fft_phase': pvn.surrogate_fft_phase,
}

CENTER = (1,1,1)


def shell_indices():
    corners, edges, faces = [], [], []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = (i,j,k)
                if idx == CENTER:
                    continue
                n_center = sum(v == 1 for v in idx)
                if n_center == 0:
                    corners.append(idx)
                elif n_center == 1:
                    edges.append(idx)
                elif n_center == 2:
                    faces.append(idx)
    return {'corner': corners, 'edge': edges, 'face': faces}

SHELLS = shell_indices()


def opposite_pairs(indices):
    idx_set = set(indices)
    seen = set()
    pairs = []
    for a in indices:
        b = tuple(2 - np.array(a))
        if b not in idx_set:
            continue
        key = tuple(sorted([a,b]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a,b))
    return pairs

PAIRS = {name: opposite_pairs(idxs) for name, idxs in SHELLS.items()}


def coh_signed(a, b, eps=1e-12):
    a = float(a); b = float(b)
    return 2.0 * a * b / (a*a + b*b + eps)


def coh_abs(a, b, eps=1e-12):
    return abs(coh_signed(a, b, eps=eps))


def shell_scores(xyz):
    c = float(xyz[CENTER])
    out = {}
    for shell, pairs in PAIRS.items():
        direct_abs = []
        direct_signed = []
        med_abs = []
        med_signed = []
        for a_idx, b_idx in pairs:
            a = float(xyz[a_idx])
            b = float(xyz[b_idx])
            d_s = coh_signed(a, b)
            d_a = abs(d_s)
            ac_s = coh_signed(a, c)
            bc_s = coh_signed(b, c)
            m_a = np.sqrt(abs(ac_s) * abs(bc_s))
            m_s = np.sign(ac_s * bc_s) * m_a
            direct_signed.append(d_s)
            direct_abs.append(d_a)
            med_abs.append(m_a)
            med_signed.append(m_s)
        out[f'{shell}_direct_abs'] = float(np.mean(direct_abs))
        out[f'{shell}_direct_signed'] = float(np.mean(direct_signed))
        out[f'{shell}_mediated_abs'] = float(np.mean(med_abs))
        out[f'{shell}_mediated_signed'] = float(np.mean(med_signed))
        out[f'{shell}_center_amp'] = abs(c)
    # hierarchy energies too
    out['center_abs'] = abs(c)
    out['corner_abs_mean'] = float(np.mean(np.abs([xyz[idx] for idx in SHELLS['corner']])))
    out['edge_abs_mean'] = float(np.mean(np.abs([xyz[idx] for idx in SHELLS['edge']])))
    out['face_abs_mean'] = float(np.mean(np.abs([xyz[idx] for idx in SHELLS['face']])))
    return out


def summarize_compare(real_val, sur_vals):
    sur_vals = np.asarray(sur_vals, dtype=float)
    mu = float(np.mean(sur_vals))
    sd = float(np.std(sur_vals, ddof=0))
    z = float((real_val - mu) / (sd + 1e-12))
    exceed = float(np.mean(real_val > sur_vals))
    p95 = float(np.quantile(sur_vals, 0.95))
    return mu, sd, z, exceed, p95


def pick_winner(rows, criterion_key):
    col = CRITERIA[criterion_key]
    sub = [r for r in rows if r['local_mode_index'] > 1]
    return max(sub, key=lambda r: r[col])


def plot_metric(summary, metric_prefix, out_png, title):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), sharey=True)
    shells = ['corner', 'edge', 'face']
    for ax, shell in zip(axes, shells):
        for crit in ['score_q', 'score_iso', 'q_only', 'xyz_only']:
            sub = summary[(summary['criterion'] == crit) & (summary['shell'] == shell)].sort_values('beta')
            ax.plot(sub['beta'], sub[f'{metric_prefix}_mean'], marker='o', label=crit)
        ax.axhline(0.5 if 'exceed' in metric_prefix else 0.0, color='k', linestyle=':', linewidth=1)
        ax.set_title(shell)
        ax.set_xlabel('beta')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(metric_prefix)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.05), fontsize=8)
    fig.suptitle(title, y=1.12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--betas', type=str, default='0,1,1.5,2,5,10')
    parser.add_argument('--pts-per-cell', type=int, default=5)
    parser.add_argument('--modes', type=int, default=10)
    parser.add_argument('--n-draws', type=int, default=48)
    parser.add_argument('--seed', type=int, default=20260320)
    parser.add_argument('--full-csv', type=str, default='/mnt/data/opposite_center_locking_full.csv')
    parser.add_argument('--summary-csv', type=str, default='/mnt/data/opposite_center_locking_summary.csv')
    parser.add_argument('--direct-exceed-plot', type=str, default='/mnt/data/opposite_center_locking_direct_exceed.png')
    parser.add_argument('--mediated-exceed-plot', type=str, default='/mnt/data/opposite_center_locking_mediated_exceed.png')
    parser.add_argument('--direct-z-plot', type=str, default='/mnt/data/opposite_center_locking_direct_z.png')
    parser.add_argument('--hierarchy-plot', type=str, default='/mnt/data/opposite_center_locking_hierarchy.png')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    betas = [float(x) for x in args.betas.split(',') if x.strip()]

    rows = []
    hier_rows = []
    for geom, lengths in GEOMETRIES:
        for beta in betas:
            base = pvn.solve_mode_xyz(lengths, beta, pts_per_cell=args.pts_per_cell, modes=args.modes)
            for criterion in CRITERIA:
                winner = pick_winner(base, criterion)
                real_scores = shell_scores(winner['xyz'])
                hier_rows.append({
                    'geometry': geom,
                    'beta': beta,
                    'criterion': criterion,
                    'winner_mode_index': winner['local_mode_index'],
                    'winner_q': winner['dominant_q'],
                    **{k: real_scores[k] for k in ['center_abs','corner_abs_mean','edge_abs_mean','face_abs_mean']}
                })
                sur_cache = {null_name: [] for null_name in NULLS}
                # collect surrogate scores per null and shell metric
                for null_name, fn in NULLS.items():
                    metric_lists = {}
                    for _ in range(args.n_draws):
                        sxyz = fn(winner['xyz'], rng)
                        ss = shell_scores(sxyz)
                        for k, v in ss.items():
                            metric_lists.setdefault(k, []).append(v)
                    for shell in ['corner','edge','face']:
                        for base_metric in ['direct_abs','direct_signed','mediated_abs','mediated_signed']:
                            key = f'{shell}_{base_metric}'
                            mu, sd, z, exceed, p95 = summarize_compare(real_scores[key], metric_lists[key])
                            rows.append({
                                'geometry': geom,
                                'beta': beta,
                                'criterion': criterion,
                                'winner_mode_index': winner['local_mode_index'],
                                'winner_q': winner['dominant_q'],
                                'null_type': null_name,
                                'shell': shell,
                                'metric': base_metric,
                                'real_value': real_scores[key],
                                'sur_mean': mu,
                                'sur_std': sd,
                                'z_score': z,
                                'exceed': exceed,
                                'sur_p95': p95,
                            })

    full = pd.DataFrame(rows)
    summary = full.groupby(['criterion','beta','null_type','shell','metric']).agg(
        real_mean=('real_value','mean'),
        sur_mean=('sur_mean','mean'),
        z_mean=('z_score','mean'),
        exceed_mean=('exceed','mean'),
    ).reset_index()
    full.to_csv(args.full_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    # fft-phase only plots
    sfft = summary[summary['null_type'] == 'fft_phase']
    plot_metric(sfft[sfft['metric'] == 'direct_abs'], 'exceed', args.direct_exceed_plot,
                'Opposite-pair locking exceedance vs fft-phase surrogate (direct abs)')
    plot_metric(sfft[sfft['metric'] == 'mediated_abs'], 'exceed', args.mediated_exceed_plot,
                'Center-mediated locking exceedance vs fft-phase surrogate (abs)')
    plot_metric(sfft[sfft['metric'] == 'direct_abs'], 'z', args.direct_z_plot,
                'Opposite-pair locking z-score vs fft-phase surrogate (direct abs)')

    hier = pd.DataFrame(hier_rows)
    hs = hier.groupby(['criterion','beta']).agg(
        center_abs_mean=('center_abs','mean'),
        corner_abs_mean=('corner_abs_mean','mean'),
        edge_abs_mean=('edge_abs_mean','mean'),
        face_abs_mean=('face_abs_mean','mean'),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8.5,5.0))
    for crit in ['score_q','score_iso','q_only','xyz_only']:
        sub = hs[hs['criterion']==crit].sort_values('beta')
        ax.plot(sub['beta'], sub['corner_abs_mean'], marker='o', label=f'{crit}: corner')
        ax.plot(sub['beta'], sub['edge_abs_mean'], marker='s', linestyle='--', label=f'{crit}: edge')
        ax.plot(sub['beta'], sub['face_abs_mean'], marker='^', linestyle=':', label=f'{crit}: face')
    ax.set_xlabel('beta')
    ax.set_ylabel('mean abs shell amplitude')
    ax.set_title('Corner / Edge / Face mean amplitude hierarchy of winners')
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.hierarchy_plot, dpi=160)
    plt.close(fig)

    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
