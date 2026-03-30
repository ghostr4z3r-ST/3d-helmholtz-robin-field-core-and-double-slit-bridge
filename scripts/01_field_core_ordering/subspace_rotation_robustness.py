import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

def _load_repo_module(name: str, relpath: str):
    local = REPO_ROOT / relpath
    fallback = Path('/mnt/data') / Path(relpath).name
    target = local if local.exists() else fallback
    spec = importlib.util.spec_from_file_location(name, str(target))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

gpm = _load_repo_module('gpm', 'scripts/01_field_core_ordering/geometry_phase_map.py')

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
}


def q_family(label: str) -> str:
    if label == 'const':
        return 'const'
    parts = label.split('+')
    vals = []
    for p in parts:
        m = re.match(r'([XYZ])(\d+)', p)
        vals.append(int(m.group(2)))
    vals = tuple(sorted(vals))
    return f'{len(parts)}-axis:{vals}'


def random_orthogonal(k: int, rng: np.random.Generator) -> np.ndarray:
    M = rng.normal(size=(k, k))
    Q, R = np.linalg.qr(M)
    s = np.sign(np.diag(R))
    s[s == 0] = 1.0
    return Q * s


def find_clusters(vals: np.ndarray, abs_tol: float = 1e-8, rel_tol: float = 1e-8):
    clusters = []
    start = 0
    for i in range(1, len(vals)):
        gap = abs(vals[i] - vals[i - 1])
        scale = max(1.0, abs(vals[i]), abs(vals[i - 1]))
        if gap <= max(abs_tol, rel_tol * scale):
            continue
        clusters.append(np.arange(start, i))
        start = i
    clusters.append(np.arange(start, len(vals)))
    return clusters


def solve_case(lengths, beta, ncell=3, pts_per_cell=5, modes=20):
    Lx, Ly, Lz = lengths
    Nx = ncell * pts_per_cell + 1
    Ny = ncell * pts_per_cell + 1
    Nz = ncell * pts_per_cell + 1
    A, hx, hy, hz = gpm.laplacian_3d(Nx, Ny, Nz, ncell * Lx, ncell * Ly, ncell * Lz, beta)
    vals, vecs = spla.eigsh(A, k=modes, which='SM', tol=1e-6)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    return vals, vecs, (Nx, Ny, Nz, hx, hy, hz)


def metric_table(vecs: np.ndarray, info, ncell=3, pts_per_cell=5) -> pd.DataFrame:
    Nx, Ny, Nz, hx, hy, hz = info
    rows = []
    for idx in range(1, vecs.shape[1]):  # skip trivial lowest mode
        u = vecs[:, idx].reshape((Nx, Ny, Nz))
        xyz = gpm.local_xyz_array(u, ncell=ncell, pts_per_cell=pts_per_cell)
        fft = np.fft.fftn(xyz)
        power = np.abs(fft) ** 2
        power[0, 0, 0] = 0.0
        q = np.unravel_index(np.argmax(power), power.shape)
        qlab = gpm.q_label(q)
        q_contrast = float(power[q] / (power.sum() + 1e-12))
        mean_abs_xyz = float(np.mean(np.abs(xyz)))
        anis = gpm.density_anisotropy(u, hx, hy, hz)
        rows.append({
            'local_mode_index': idx + 1,
            'dominant_q': qlab,
            'q_family': q_family(qlab),
            'score_q': mean_abs_xyz * q_contrast,
            'score_iso': mean_abs_xyz * anis,
        })
    return pd.DataFrame(rows)


def rotate_subspaces(vecs: np.ndarray, clusters, rng: np.random.Generator) -> np.ndarray:
    out = vecs.copy()
    for cl in clusters:
        if len(cl) > 1:
            out[:, cl] = out[:, cl] @ random_orthogonal(len(cl), rng)
    return out


def modal_fraction(seq):
    vals, counts = np.unique(seq, return_counts=True)
    idx = np.argmax(counts)
    return vals[idx], int(counts[idx]), float(counts[idx] / len(seq))


def plot_heatmap(df, value_col, text_col, out_png, title):
    geoms = list(df['geometry'].drop_duplicates())
    betas = list(df['beta'].drop_duplicates())
    M = np.full((len(geoms), len(betas)), np.nan)
    T = [['' for _ in betas] for _ in geoms]
    for i, g in enumerate(geoms):
        for j, b in enumerate(betas):
            sub = df[(df.geometry == g) & (df.beta == b)]
            if len(sub):
                row = sub.iloc[0]
                M[i, j] = row[value_col]
                T[i][j] = row[text_col]
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(M, aspect='auto', vmin=0.0, vmax=1.0, cmap='viridis')
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([str(b) for b in betas])
    ax.set_yticks(range(len(geoms)))
    ax.set_yticklabels(geoms)
    ax.set_xlabel('beta')
    ax.set_title(title)
    for i in range(len(geoms)):
        for j in range(len(betas)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, T[i][j], ha='center', va='center', color='white', fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--betas', type=str, default='0,0.25,0.5,0.75,1,1.5,2,3,4,5,7,10')
    parser.add_argument('--pts-per-cell', type=int, default=5)
    parser.add_argument('--modes', type=int, default=20)
    parser.add_argument('--trials', type=int, default=24)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--full-csv', type=str, default='/mnt/data/subspace_rotation_robustness_full.csv')
    parser.add_argument('--summary-csv', type=str, default='/mnt/data/subspace_rotation_robustness_summary.csv')
    parser.add_argument('--heatmap-png', type=str, default='/mnt/data/subspace_rotation_robustness_consensus_family.png')
    parser.add_argument('--q-heatmap-png', type=str, default='/mnt/data/subspace_rotation_robustness_consensus_q.png')
    args = parser.parse_args()

    betas = [float(x) for x in args.betas.split(',') if x.strip()]
    rng = np.random.default_rng(args.seed)

    full_rows = []
    summary_rows = []
    for gname, lengths in GEOMETRIES:
        for beta in betas:
            vals, vecs, info = solve_case(lengths, beta, pts_per_cell=args.pts_per_cell, modes=args.modes)
            clusters = find_clusters(vals)
            deg_clusters = [c for c in clusters if len(c) > 1]
            max_cluster = max([len(c) for c in clusters], default=1)
            for trial in range(args.trials):
                trial_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                V = rotate_subspaces(vecs, clusters, trial_rng)
                mt = metric_table(V, info, pts_per_cell=args.pts_per_cell)
                winners = {}
                for cname, col in CRITERIA.items():
                    best = mt.sort_values(col, ascending=False).iloc[0]
                    winners[cname] = {
                        'dominant_q': best['dominant_q'],
                        'q_family': best['q_family'],
                    }
                    full_rows.append({
                        'geometry': gname,
                        'beta': beta,
                        'trial': trial,
                        'criterion': cname,
                        'dominant_q': best['dominant_q'],
                        'q_family': best['q_family'],
                        'num_deg_clusters': len(deg_clusters),
                        'max_cluster_size': max_cluster,
                    })
                fams = [winners[c]['q_family'] for c in CRITERIA]
                qs = [winners[c]['dominant_q'] for c in CRITERIA]
                fam_mode, _, _ = modal_fraction(fams)
                q_mode, _, _ = modal_fraction(qs)
                full_rows.append({
                    'geometry': gname,
                    'beta': beta,
                    'trial': trial,
                    'criterion': 'consensus',
                    'dominant_q': q_mode,
                    'q_family': fam_mode,
                    'num_deg_clusters': len(deg_clusters),
                    'max_cluster_size': max_cluster,
                })

            # summarize across trials
            trial_df = pd.DataFrame([r for r in full_rows if r['geometry'] == gname and r['beta'] == beta])
            for criterion in ['score_q', 'score_iso', 'consensus']:
                sub = trial_df[trial_df.criterion == criterion]
                fam_mode, fam_count, fam_frac = modal_fraction(sub['q_family'].tolist())
                q_mode, q_count, q_frac = modal_fraction(sub['dominant_q'].tolist())
                summary_rows.append({
                    'geometry': gname,
                    'beta': beta,
                    'criterion': criterion,
                    'num_deg_clusters': len(deg_clusters),
                    'max_cluster_size': max_cluster,
                    'family_mode': fam_mode,
                    'family_mode_count': fam_count,
                    'family_mode_frac': fam_frac,
                    'exact_q_mode': q_mode,
                    'exact_q_mode_count': q_count,
                    'exact_q_mode_frac': q_frac,
                    'n_unique_families': int(sub['q_family'].nunique()),
                    'n_unique_exact_q': int(sub['dominant_q'].nunique()),
                })

    full_df = pd.DataFrame(full_rows)
    summary_df = pd.DataFrame(summary_rows)
    full_df.to_csv(args.full_csv, index=False)
    summary_df.to_csv(args.summary_csv, index=False)

    cons = summary_df[summary_df.criterion == 'consensus'].copy()
    cons['label'] = cons['family_mode'] + '\n' + cons['family_mode_frac'].map(lambda x: f'{x:.2f}')
    plot_heatmap(cons, 'family_mode_frac', 'label', args.heatmap_png,
                 'Subspace-rotation robustness: consensus family mode')
    consq = summary_df[summary_df.criterion == 'consensus'].copy()
    consq['label'] = consq['exact_q_mode'] + '\n' + consq['exact_q_mode_frac'].map(lambda x: f'{x:.2f}')
    plot_heatmap(consq, 'exact_q_mode_frac', 'label', args.q_heatmap_png,
                 'Subspace-rotation robustness: consensus exact q label')


if __name__ == '__main__':
    main()
