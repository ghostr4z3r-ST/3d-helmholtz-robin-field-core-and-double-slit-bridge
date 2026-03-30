import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import importlib.util

spec = importlib.util.spec_from_file_location('gpm','/mnt/data/geometry_phase_map.py')
gpm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpm)

GEOMETRIES = [
    ('cubic', (1.0, 1.0, 1.0)),
    ('tet_1.05', (1.0, 1.0, 1.05)),
    ('tet_1.10', (1.0, 1.0, 1.10)),
    ('ortho_1.05_1.20', (1.0, 1.05, 1.20)),
    ('tet_1.20', (1.0, 1.0, 1.20)),
    ('ortho_1.10_1.30', (1.0, 1.10, 1.30)),
    ('tet_1.35', (1.0, 1.0, 1.35)),
    ('ortho_1.20_1.50', (1.0, 1.20, 1.50)),
    ('tet_1.50', (1.0, 1.0, 1.50)),
]

CRITERIA = {
    'score_q': 'score_q',
    'score_iso': 'score_iso',
    'q_only': 'score_q_only',
    'xyz_only': 'score_xyz_only',
}

VARIANTS = [
    ('base_r1', (0, 0, 0), 'r1'),
    ('x+_r1', (1, 0, 0), 'r1'),
    ('y+_r1', (0, 1, 0), 'r1'),
    ('z+_r1', (0, 0, 1), 'r1'),
    ('xy+_r1', (1, 1, 0), 'r1'),
    ('xz+_r1', (1, 0, 1), 'r1'),
    ('yz+_r1', (0, 1, 1), 'r1'),
    ('xyz+_r1', (1, 1, 1), 'r1'),
    ('base_r2', (0, 0, 0), 'r2'),
    ('base_avg12', (0, 0, 0), 'avg12'),
]


def q_family(label: str) -> str:
    if label == 'const':
        return 'const'
    vals = []
    for part in label.split('+'):
        m = re.match(r'([XYZ])(\d+)', part)
        vals.append(int(m.group(2)))
    vals = tuple(sorted(vals))
    return f"{len(vals)}-axis:{vals}"


def local_xyz_array_variant(u, ncell, pts_per_cell, shift=(0, 0, 0), mode='r1'):
    def one_radius(radius, center_shift):
        arr = np.zeros((ncell, ncell, ncell), dtype=float)
        offs = [
            (-radius, -radius, -radius), (-radius, -radius, radius),
            (-radius, radius, -radius), (-radius, radius, radius),
            (radius, -radius, -radius), (radius, -radius, radius),
            (radius, radius, -radius), (radius, radius, radius),
        ]
        signs = np.array([a * b * c for a, b, c in offs], dtype=float)
        signs = np.sign(signs)
        for i in range(ncell):
            for j in range(ncell):
                for k in range(ncell):
                    cx = i * pts_per_cell + pts_per_cell // 2 + center_shift[0]
                    cy = j * pts_per_cell + pts_per_cell // 2 + center_shift[1]
                    cz = k * pts_per_cell + pts_per_cell // 2 + center_shift[2]
                    vals = np.array([u[cx + a, cy + b, cz + c] for a, b, c in offs], dtype=float)
                    arr[i, j, k] = np.mean(signs * vals)
        return arr

    if mode == 'r1':
        return one_radius(1, shift)
    if mode == 'r2':
        return one_radius(2, shift)
    if mode == 'avg12':
        return 0.5 * (one_radius(1, shift) + one_radius(2, shift))
    raise ValueError(mode)


def solve_fields(lengths, beta, ncell=3, pts_per_cell=5, modes=10):
    Lx, Ly, Lz = lengths
    Nx = ncell * pts_per_cell + 1
    Ny = ncell * pts_per_cell + 1
    Nz = ncell * pts_per_cell + 1
    A, hx, hy, hz = gpm.laplacian_3d(Nx, Ny, Nz, ncell * Lx, ncell * Ly, ncell * Lz, beta)
    vals, vecs = spla.eigsh(A, k=modes, which='SM', tol=1e-6)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    fields = []
    for idx in range(modes):
        u = vecs[:, idx].reshape((Nx, Ny, Nz))
        anis = gpm.density_anisotropy(u, hx, hy, hz)
        B = gpm.boundary_ratio(u, hx, hy, hz)
        fields.append((idx + 1, float(vals[idx]), u, anis, B))
    return fields


def mode_variant_table(lengths, beta, ncell=3, pts_per_cell=5, modes=10):
    fields = solve_fields(lengths, beta, ncell=ncell, pts_per_cell=pts_per_cell, modes=modes)
    rows = []
    for mode_index, eigval, u, anis, B in fields:
        for vname, shift, vmode in VARIANTS:
            xyz = local_xyz_array_variant(u, ncell=ncell, pts_per_cell=pts_per_cell, shift=shift, mode=vmode)
            fft = np.fft.fftn(xyz)
            power = np.abs(fft) ** 2
            power[0, 0, 0] = 0.0
            q = np.unravel_index(np.argmax(power), power.shape)
            q_contrast = float(power[q] / (power.sum() + 1e-12))
            mean_abs_xyz = float(np.mean(np.abs(xyz)))
            rows.append({
                'local_mode_index': mode_index,
                'eigenvalue': eigval,
                'variant': vname,
                'dominant_q': gpm.q_label(q),
                'q_family': q_family(gpm.q_label(q)),
                'q_contrast': q_contrast,
                'mean_abs_xyz': mean_abs_xyz,
                'field_anisotropy': anis,
                'boundary_ratio': B,
                'score_q': mean_abs_xyz * q_contrast,
                'score_iso': mean_abs_xyz * anis,
                'score_q_only': q_contrast,
                'score_xyz_only': mean_abs_xyz,
            })
    return pd.DataFrame(rows)


def consensus_label(series):
    counts = series.value_counts().sort_index()
    max_count = counts.max()
    return sorted(counts[counts == max_count].index)[0], int(max_count)


def plot_heatmap(summary, value_col, title, out_png, fmt='{:.2f}', cmap='viridis', vmin=0.0, vmax=1.0):
    geoms = list(summary['geometry'].drop_duplicates())
    betas = list(summary['beta'].drop_duplicates())
    M = np.full((len(geoms), len(betas)), np.nan)
    for i, g in enumerate(geoms):
        for j, b in enumerate(betas):
            sub = summary[(summary.geometry == g) & (summary.beta == b)]
            if len(sub):
                M[i, j] = sub.iloc[0][value_col]
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(M, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([str(b) for b in betas])
    ax.set_yticks(range(len(geoms)))
    ax.set_yticklabels(geoms)
    ax.set_xlabel('beta')
    ax.set_title(title)
    for i in range(len(geoms)):
        for j in range(len(betas)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center', color='white', fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    # radius-1 local readout needs a central point with one neighbor in each direction
    parser.add_argument('--betas', type=str, default='0,1,1.5,2,5,10')
    parser.add_argument('--pts-per-cell', type=int, default=5)
    parser.add_argument('--modes', type=int, default=10)
    parser.add_argument('--full-csv', type=str, default='/mnt/data/readout_robustness_full.csv')
    parser.add_argument('--winners-csv', type=str, default='/mnt/data/readout_robustness_winners.csv')
    parser.add_argument('--summary-csv', type=str, default='/mnt/data/readout_robustness_summary.csv')
    parser.add_argument('--family-plot', type=str, default='/mnt/data/readout_robustness_family.png')
    parser.add_argument('--q-plot', type=str, default='/mnt/data/readout_robustness_q.png')
    args = parser.parse_args()

    betas = [float(x) for x in args.betas.split(',') if x.strip()]
    frames = []
    for name, lengths in GEOMETRIES:
        for beta in betas:
            df = mode_variant_table(lengths, beta, pts_per_cell=args.pts_per_cell, modes=args.modes)
            df.insert(0, 'beta', beta)
            df.insert(0, 'geometry', name)
            frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(args.full_csv, index=False)

    work = full[full.local_mode_index > 1].copy()
    winner_rows = []
    for (geom, beta, variant), grp in work.groupby(['geometry', 'beta', 'variant']):
        for cname, col in CRITERIA.items():
            best = grp.sort_values(col, ascending=False).iloc[0]
            winner_rows.append({
                'geometry': geom,
                'beta': beta,
                'variant': variant,
                'criterion': cname,
                'local_mode_index': int(best.local_mode_index),
                'eigenvalue': best.eigenvalue,
                'dominant_q': best.dominant_q,
                'q_family': best.q_family,
                'q_contrast': best.q_contrast,
                'mean_abs_xyz': best.mean_abs_xyz,
                'field_anisotropy': best.field_anisotropy,
                'boundary_ratio': best.boundary_ratio,
                'criterion_value': best[col],
            })
    winners = pd.DataFrame(winner_rows)
    winners.to_csv(args.winners_csv, index=False)

    rows = []
    for (geom, beta, criterion), grp in winners.groupby(['geometry', 'beta', 'criterion']):
        fam_cons, fam_count = consensus_label(grp['q_family'])
        q_cons, q_count = consensus_label(grp['dominant_q'])
        rows.append({
            'geometry': geom,
            'beta': beta,
            'criterion': criterion,
            'family_consensus': fam_cons,
            'family_consensus_count': fam_count,
            'family_match_frac': float((grp['q_family'] == fam_cons).mean()),
            'family_unique': int(grp['q_family'].nunique()),
            'q_consensus': q_cons,
            'q_consensus_count': q_count,
            'exact_q_match_frac': float((grp['dominant_q'] == q_cons).mean()),
            'q_unique': int(grp['dominant_q'].nunique()),
            'winner_mode_unique': int(grp['local_mode_index'].nunique()),
        })
    percrit = pd.DataFrame(rows)

    summary = percrit.groupby(['geometry', 'beta']).agg(
        family_match_frac=('family_match_frac', 'mean'),
        exact_q_match_frac=('exact_q_match_frac', 'mean'),
        mean_family_unique=('family_unique', 'mean'),
        mean_q_unique=('q_unique', 'mean'),
        mean_winner_mode_unique=('winner_mode_unique', 'mean'),
    ).reset_index()
    summary.to_csv(args.summary_csv, index=False)

    plot_heatmap(summary, 'family_match_frac', 'Readout robustness: family match fraction', args.family_plot)
    plot_heatmap(summary, 'exact_q_match_frac', 'Readout robustness: exact q match fraction', args.q_plot)

    print('\nPer-geometry/beta summary:\n')
    print(summary.to_string(index=False))
    print('\nCriterion-level detail:\n')
    print(percrit.to_string(index=False))


if __name__ == '__main__':
    main()
