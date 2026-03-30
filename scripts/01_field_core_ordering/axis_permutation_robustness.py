import argparse
import itertools
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    'q_only': 'score_q_only',
    'xyz_only': 'score_xyz_only',
}
AXES = 'XYZ'


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


def unique_permutations(lengths):
    seen = {}
    for perm in itertools.permutations(range(3)):
        perm_lengths = tuple(lengths[i] for i in perm)
        if perm_lengths not in seen:
            seen[perm_lengths] = perm
    # sort deterministic by lengths then perm tuple
    items = [(perm_lengths, perm) for perm_lengths, perm in seen.items()]
    items.sort(key=lambda x: (x[0], x[1]))
    return items


def invert_perm(perm):
    inv = [None] * len(perm)
    for new_idx, old_idx in enumerate(perm):
        inv[old_idx] = new_idx
    return tuple(inv)


def permute_q_label(label: str, perm) -> str:
    if label == 'const':
        return label
    inv = invert_perm(perm)
    parts = []
    for p in label.split('+'):
        m = re.match(r'([XYZ])(\d+)', p)
        old_axis = AXES.index(m.group(1))
        mag = m.group(2)
        new_axis = AXES[inv[old_axis]]
        parts.append((AXES.index(new_axis), f'{new_axis}{mag}'))
    parts.sort(key=lambda t: t[0])
    return '+'.join(x[1] for x in parts)


def consensus_mode(labels):
    c = Counter(labels)
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def solve_metric_table(lengths, beta, ncell=3, pts_per_cell=5, modes=20):
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
    for idx in range(1, modes):  # skip trivial lowest mode
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
            'eigenvalue': float(vals[idx]),
            'dominant_q': qlab,
            'q_family': q_family(qlab),
            'score_q': mean_abs_xyz * q_contrast,
            'score_iso': mean_abs_xyz * anis,
            'score_q_only': q_contrast,
            'score_xyz_only': mean_abs_xyz,
        })
    return pd.DataFrame(rows)


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
    parser.add_argument('--betas', type=str, default='0,0.5,1,1.5,2,5,10')
    parser.add_argument('--pts-per-cell', type=int, default=5)
    parser.add_argument('--modes', type=int, default=20)
    parser.add_argument('--full-csv', type=str, default='/mnt/data/axis_permutation_robustness_full.csv')
    parser.add_argument('--summary-csv', type=str, default='/mnt/data/axis_permutation_robustness_summary.csv')
    parser.add_argument('--family-heatmap-png', type=str, default='/mnt/data/axis_permutation_robustness_consensus_family.png')
    parser.add_argument('--q-heatmap-png', type=str, default='/mnt/data/axis_permutation_robustness_consensus_q.png')
    args = parser.parse_args()

    betas = [float(x) for x in args.betas.split(',') if x.strip()]
    cache = {}

    def get_table(lengths, beta):
        key = (tuple(float(x) for x in lengths), float(beta), args.pts_per_cell, args.modes)
        if key not in cache:
            cache[key] = solve_metric_table(lengths, beta, pts_per_cell=args.pts_per_cell, modes=args.modes)
        return cache[key]

    full_rows = []
    summary_rows = []

    for gname, lengths in GEOMETRIES:
        perms = unique_permutations(lengths)
        for beta in betas:
            base = get_table(lengths, beta)
            baseline = {}
            for cname, col in CRITERIA.items():
                best = base.sort_values(col, ascending=False).iloc[0]
                baseline[cname] = {
                    'q': best['dominant_q'],
                    'family': best['q_family'],
                }
            baseline['consensus'] = {
                'q': consensus_mode([baseline[c]['q'] for c in CRITERIA]),
                'family': consensus_mode([baseline[c]['family'] for c in CRITERIA]),
            }

            for perm_lengths, perm in perms:
                tag = ''.join(AXES[i] for i in perm)
                tab = get_table(perm_lengths, beta)
                obs = {}
                for cname, col in CRITERIA.items():
                    best = tab.sort_values(col, ascending=False).iloc[0]
                    obs[cname] = {
                        'q': best['dominant_q'],
                        'family': best['q_family'],
                    }
                obs['consensus'] = {
                    'q': consensus_mode([obs[c]['q'] for c in CRITERIA]),
                    'family': consensus_mode([obs[c]['family'] for c in CRITERIA]),
                }

                for cname in list(CRITERIA.keys()) + ['consensus']:
                    expected_q = permute_q_label(baseline[cname]['q'], perm)
                    expected_family = baseline[cname]['family']
                    observed_q = obs[cname]['q']
                    observed_family = obs[cname]['family']
                    full_rows.append({
                        'geometry': gname,
                        'beta': beta,
                        'perm_tag': tag,
                        'perm': str(perm),
                        'criterion': cname,
                        'baseline_q': baseline[cname]['q'],
                        'expected_q': expected_q,
                        'observed_q': observed_q,
                        'exact_match': int(expected_q == observed_q),
                        'baseline_family': expected_family,
                        'observed_family': observed_family,
                        'family_match': int(expected_family == observed_family),
                    })

            sub = pd.DataFrame([r for r in full_rows if r['geometry'] == gname and r['beta'] == beta])
            for cname in list(CRITERIA.keys()) + ['consensus']:
                s = sub[sub.criterion == cname]
                summary_rows.append({
                    'geometry': gname,
                    'beta': beta,
                    'criterion': cname,
                    'n_permutations': int(len(s)),
                    'family_match_frac': float(s['family_match'].mean()),
                    'exact_q_match_frac': float(s['exact_match'].mean()),
                    'n_unique_observed_families': int(s['observed_family'].nunique()),
                    'n_unique_observed_q': int(s['observed_q'].nunique()),
                    'family_mode': consensus_mode(s['observed_family'].tolist()),
                    'exact_q_mode': consensus_mode(s['observed_q'].tolist()),
                })

    full_df = pd.DataFrame(full_rows)
    summary_df = pd.DataFrame(summary_rows)
    full_df.to_csv(args.full_csv, index=False)
    summary_df.to_csv(args.summary_csv, index=False)

    cons = summary_df[summary_df.criterion == 'consensus'].copy()
    cons['label'] = cons['family_mode'] + '\n' + cons['family_match_frac'].map(lambda x: f'{x:.2f}')
    plot_heatmap(cons, 'family_match_frac', 'label', args.family_heatmap_png,
                 'Axis-permutation robustness: consensus family covariance')
    consq = summary_df[summary_df.criterion == 'consensus'].copy()
    consq['label'] = consq['exact_q_mode'] + '\n' + consq['exact_q_match_frac'].map(lambda x: f'{x:.2f}')
    plot_heatmap(consq, 'exact_q_match_frac', 'label', args.q_heatmap_png,
                 'Axis-permutation robustness: consensus exact-q covariance')


if __name__ == '__main__':
    main()
