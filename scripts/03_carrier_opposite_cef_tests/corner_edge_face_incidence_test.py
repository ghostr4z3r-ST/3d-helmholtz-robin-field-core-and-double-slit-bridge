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
                idx = (i, j, k)
                if idx == CENTER:
                    continue
                n_center = sum(v == 1 for v in idx)
                if n_center == 0:
                    corners.append(idx)
                elif n_center == 1:
                    edges.append(idx)
                elif n_center == 2:
                    faces.append(idx)
    return corners, edges, faces

CORNERS, EDGES, FACES = shell_indices()


def edge_incident_corners(e):
    axis = [i for i, v in enumerate(e) if v == 1][0]
    a = list(e); b = list(e)
    a[axis] = 0; b[axis] = 2
    return [tuple(a), tuple(b)]


def face_incident_edges(f):
    fixed_axes = [i for i, v in enumerate(f) if v != 1]
    fixed_axis = fixed_axes[0]
    fixed_val = f[fixed_axis]
    other_axes = [0,1,2]
    other_axes.remove(fixed_axis)
    edges = []
    for ax in other_axes:
        for val in [0,2]:
            e = [1,1,1]
            e[fixed_axis] = fixed_val
            e[ax] = val
            edges.append(tuple(e))
    return edges


def face_incident_corners(f):
    fixed_axes = [i for i, v in enumerate(f) if v != 1]
    fixed_axis = fixed_axes[0]
    fixed_val = f[fixed_axis]
    other_axes = [0,1,2]
    other_axes.remove(fixed_axis)
    corners = []
    for v1 in [0,2]:
        for v2 in [0,2]:
            c = [0,0,0]
            c[fixed_axis] = fixed_val
            c[other_axes[0]] = v1
            c[other_axes[1]] = v2
            corners.append(tuple(c))
    return corners


EDGE_TO_CORNERS = {e: edge_incident_corners(e) for e in EDGES}
FACE_TO_EDGES = {f: face_incident_edges(f) for f in FACES}
FACE_TO_CORNERS = {f: face_incident_corners(f) for f in FACES}


def vec_abs_corr(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(abs(np.dot(a, b) / (na * nb)))


def vec_signed_corr(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def incidence_scores(xyz):
    edge_vals = np.array([xyz[e] for e in EDGES], dtype=float)
    edge_pred_corner = np.array([np.mean([xyz[c] for c in EDGE_TO_CORNERS[e]]) for e in EDGES], dtype=float)
    face_vals = np.array([xyz[f] for f in FACES], dtype=float)
    face_pred_edge = np.array([np.mean([xyz[e] for e in FACE_TO_EDGES[f]]) for f in FACES], dtype=float)
    face_pred_corner = np.array([np.mean([xyz[c] for c in FACE_TO_CORNERS[f]]) for f in FACES], dtype=float)

    out = {
        'edge_from_corner_abs': vec_abs_corr(edge_vals, edge_pred_corner),
        'edge_from_corner_signed': vec_signed_corr(edge_vals, edge_pred_corner),
        'face_from_edge_abs': vec_abs_corr(face_vals, face_pred_edge),
        'face_from_edge_signed': vec_signed_corr(face_vals, face_pred_edge),
        'face_from_corner_abs': vec_abs_corr(face_vals, face_pred_corner),
        'face_from_corner_signed': vec_signed_corr(face_vals, face_pred_corner),
    }
    out['hierarchy_gap_abs'] = 0.5 * (out['edge_from_corner_abs'] + out['face_from_edge_abs']) - out['face_from_corner_abs']
    out['hierarchy_gap_signed'] = 0.5 * (out['edge_from_corner_signed'] + out['face_from_edge_signed']) - out['face_from_corner_signed']
    out['edge_vs_facecorner_abs'] = out['edge_from_corner_abs'] - out['face_from_corner_abs']
    out['faceedge_vs_facecorner_abs'] = out['face_from_edge_abs'] - out['face_from_corner_abs']
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--betas', type=str, default='0,1,1.5,2,5,10')
    parser.add_argument('--pts-per-cell', type=int, default=5)
    parser.add_argument('--modes', type=int, default=10)
    parser.add_argument('--n-draws', type=int, default=40)
    parser.add_argument('--seed', type=int, default=20260320)
    parser.add_argument('--full-csv', type=str, default='/mnt/data/corner_edge_face_incidence_full.csv')
    parser.add_argument('--summary-csv', type=str, default='/mnt/data/corner_edge_face_incidence_summary.csv')
    parser.add_argument('--fft-exceed-plot', type=str, default='/mnt/data/corner_edge_face_incidence_fft_exceed.png')
    parser.add_argument('--overview-plot', type=str, default='/mnt/data/corner_edge_face_incidence_overview.png')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    betas = [float(x) for x in args.betas.split(',') if x.strip()]
    rows = []
    for geom, lengths in GEOMETRIES:
        for beta in betas:
            base = pvn.solve_mode_xyz(lengths, beta, pts_per_cell=args.pts_per_cell, modes=args.modes)
            for criterion in CRITERIA:
                winner = pick_winner(base, criterion)
                real_scores = incidence_scores(winner['xyz'])
                for null_name, fn in NULLS.items():
                    sur = {k: [] for k in real_scores}
                    for _ in range(args.n_draws):
                        sxyz = fn(winner['xyz'], rng)
                        ss = incidence_scores(sxyz)
                        for k, v in ss.items():
                            sur[k].append(v)
                    for metric, real_val in real_scores.items():
                        mu, sd, z, exceed, p95 = summarize_compare(real_val, sur[metric])
                        rows.append({
                            'geometry': geom,
                            'beta': beta,
                            'criterion': criterion,
                            'winner_mode_index': winner['local_mode_index'],
                            'winner_q': winner['dominant_q'],
                            'null_type': null_name,
                            'metric': metric,
                            'real_value': real_val,
                            'sur_mean': mu,
                            'sur_std': sd,
                            'z_score': z,
                            'exceed': exceed,
                            'sur_p95': p95,
                        })
    full = pd.DataFrame(rows)
    summary = full.groupby(['criterion','beta','null_type','metric']).agg(
        real_mean=('real_value','mean'),
        sur_mean=('sur_mean','mean'),
        z_mean=('z_score','mean'),
        exceed_mean=('exceed','mean'),
    ).reset_index()
    full.to_csv(args.full_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    sfft = summary[summary['null_type'] == 'fft_phase']
    metrics = ['edge_from_corner_abs','face_from_edge_abs','face_from_corner_abs','hierarchy_gap_abs']
    fig, axes = plt.subplots(2,2, figsize=(11,8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        for crit in ['score_q','score_iso','q_only','xyz_only']:
            sub = sfft[(sfft['criterion']==crit) & (sfft['metric']==metric)].sort_values('beta')
            ax.plot(sub['beta'], sub['exceed_mean'], marker='o', label=crit)
        ax.axhline(0.5, color='k', linestyle=':', linewidth=1)
        ax.set_title(metric)
        ax.set_xlabel('beta')
        ax.set_ylabel('exceedance')
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='upper center', bbox_to_anchor=(0.5,1.02), fontsize=8)
    fig.tight_layout()
    fig.savefig(args.fft_exceed_plot, dpi=160, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2,2, figsize=(11,8), sharex=True)
    axes = axes.ravel()
    for ax, metric in zip(axes, metrics):
        sub = sfft[sfft['metric']==metric].groupby('beta', as_index=False).agg(real_mean=('real_mean','mean'), sur_mean=('sur_mean','mean'))
        ax.plot(sub['beta'], sub['real_mean'], marker='o', label='real')
        ax.plot(sub['beta'], sub['sur_mean'], marker='s', linestyle='--', label='fft-phase mean')
        ax.set_title(metric)
        ax.set_xlabel('beta')
        ax.set_ylabel('score')
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='upper center', bbox_to_anchor=(0.5,1.02))
    fig.tight_layout()
    fig.savefig(args.overview_plot, dpi=160, bbox_inches='tight')
    plt.close(fig)

    print(summary.to_string(index=False))

if __name__ == '__main__':
    main()
