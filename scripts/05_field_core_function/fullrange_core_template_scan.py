import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ov = load_module('ov', '/mnt/data/overlay_bridge_and_shell_tests.py')
pvn = load_module('pvn', '/mnt/data/phase_vs_nullmodels.py')

BETAS = [0.0, 1.0, 1.5, 2.0, 5.0, 10.0]
CRITS = list(ov.CRITERIA.keys())
GEOMETRIES = pvn.GEOMETRIES
TARGETS = np.round(np.arange(0.05, 0.951, 0.01), 3)
CENTER = (1, 1, 1)

CANON = []
SECTOR_META = []
for i in range(3):
    for j in range(3):
        for k in range(3):
            if (i, j, k) == CENTER:
                continue
            d = np.array([i - 1, j - 1, k - 1], dtype=float)
            d /= np.linalg.norm(d)
            CANON.append(d)
            n_center = sum(v == 1 for v in (i, j, k))
            shell = {0: 'corner', 1: 'edge', 2: 'face'}[n_center]
            SECTOR_META.append(((i, j, k), shell))
CANON = np.array(CANON, dtype=float)
face_idxs = [idx for idx, meta in enumerate(SECTOR_META) if meta[1] == 'face']
face_map = {}
for idx, ((i, j, k), shell) in enumerate(SECTOR_META):
    if shell == 'face':
        face_map[(i - 1, j - 1, k - 1)] = idx

plane_sets = {
    'xy': [face_map[(-1, 0, 0)], face_map[(1, 0, 0)], face_map[(0, -1, 0)], face_map[(0, 1, 0)]],
    'xz': [face_map[(-1, 0, 0)], face_map[(1, 0, 0)], face_map[(0, 0, -1)], face_map[(0, 0, 1)]],
    'yz': [face_map[(0, -1, 0)], face_map[(0, 1, 0)], face_map[(0, 0, -1)], face_map[(0, 0, 1)]],
}
axis_pairs = {
    'x': [face_map[(-1, 0, 0)], face_map[(1, 0, 0)]],
    'y': [face_map[(0, -1, 0)], face_map[(0, 1, 0)]],
    'z': [face_map[(0, 0, -1)], face_map[(0, 0, 1)]],
}


def soft_sector_vec(coords, weights, center, kappa=8.0):
    d = coords - center
    n = np.linalg.norm(d, axis=1)
    keep = n > 1e-9
    if not np.any(keep):
        return np.zeros(len(CANON))
    d = d[keep] / n[keep, None]
    w = weights[keep]
    sims = d @ CANON.T
    W = np.exp(kappa * sims)
    W /= W.sum(axis=1, keepdims=True)
    vec = (w[:, None] * W).sum(axis=0)
    vec /= vec.sum() + 1e-12
    return vec


def effective_occupancy(p):
    s = p.sum()
    if s <= 1e-12:
        return 0.0
    q = p / s
    H = -(q[q > 1e-12] * np.log(q[q > 1e-12])).sum()
    return float(np.exp(H) / len(q))


def split_inner_outer_band(u, target, bws=(0.003, 0.005, 0.007, 0.01, 0.013, 0.02), min_count=8, min_gap=0.5):
    A = np.abs(u)
    A = (A - A.min()) / (A.max() - A.min() + 1e-12)
    shp = A.shape
    coords = np.indices(shp).reshape(3, -1).T.astype(float)
    center = (np.array(shp) - 1.0) / 2.0
    vals = A.reshape(-1)
    radii = np.linalg.norm(coords - center, axis=1)
    for bw in bws:
        mask = np.abs(vals - target) <= bw
        if mask.sum() < 2 * min_count:
            continue
        rr = radii[mask]
        ww = vals[mask]
        cc = coords[mask]
        c1, c2 = np.quantile(rr, [0.25, 0.75])
        ok = True
        for _ in range(50):
            lab = np.abs(rr - c1) <= np.abs(rr - c2)
            if lab.all() or (~lab).all():
                ok = False
                break
            nc1 = rr[lab].mean()
            nc2 = rr[~lab].mean()
            if abs(nc1 - c1) + abs(nc2 - c2) < 1e-8:
                break
            c1, c2 = nc1, nc2
        if not ok:
            continue
        if c1 <= c2:
            inner, outer = lab, ~lab
            ri, ro = c1, c2
        else:
            inner, outer = ~lab, lab
            ri, ro = c2, c1
        if inner.sum() < min_count or outer.sum() < min_count or (ro - ri) < min_gap:
            continue
        return {'cc': cc, 'ww': ww, 'center': center, 'inner': inner, 'outer': outer, 'ri': ri, 'ro': ro, 'gap': ro - ri, 'bw': bw}
    return None


def shape_scores(inner_coords, inner_w, center):
    d = inner_coords - center
    n = np.linalg.norm(d, axis=1)
    keep = n > 1e-9
    d = d[keep] / n[keep, None]
    w = inner_w[keep]
    w = w / (w.sum() + 1e-12)
    M = (d[:, :, None] * d[:, None, :] * w[:, None, None]).sum(axis=0)
    eig = np.linalg.eigvalsh(M)[::-1]
    anis = np.sqrt(((eig - 1 / 3.0) ** 2).sum()) / np.sqrt(2 / 3)
    closed = float(max(0.0, 1.0 - anis))
    vec = soft_sector_vec(inner_coords, inner_w, center)
    face_w = np.array([vec[i] for i in face_idxs], dtype=float)
    face_mass = float(face_w.sum())
    axial = max(sum(vec[i] for i in idxs) for idxs in axis_pairs.values())
    square = 0.0
    for idxs in plane_sets.values():
        vals = np.array([vec[i] for i in idxs], dtype=float)
        mean = vals.mean() + 1e-12
        bal = max(0.0, 1.0 - vals.std(ddof=0) / mean)
        square = max(square, float(vals.sum() * bal))
    cube = float(face_mass * effective_occupancy(face_w))
    return {'closed': closed, 'axial': axial, 'square': square, 'cube': cube, 'eig1': eig[0], 'eig2': eig[1], 'eig3': eig[2], 'face_mass': face_mass, 'face_occ': effective_occupancy(face_w)}


rows = []
for gname, lengths in GEOMETRIES:
    for beta in BETAS:
        base_rows = ov.solve_mode_full(lengths, beta, ncell=3, pts_per_cell=5, modes=10)
        winners = ov.pick_winners(base_rows)
        for crit in CRITS:
            u = winners[crit]['u']
            for target in TARGETS:
                sp = split_inner_outer_band(u, float(target))
                if sp is None:
                    rows.append({'geometry': gname, 'beta': beta, 'criterion': crit, 'target': float(target), 'valid': False})
                    continue
                sc = shape_scores(sp['cc'][sp['inner']], sp['ww'][sp['inner']], sp['center'])
                items = sorted([(k, sc[k]) for k in ['closed', 'axial', 'square', 'cube']], key=lambda kv: kv[1], reverse=True)
                rows.append({'geometry': gname, 'beta': beta, 'criterion': crit, 'target': float(target), 'valid': True,
                             'winner': items[0][0], 'margin': items[0][1] - items[1][1],
                             'bw': sp['bw'], 'inner_r': sp['ri'], 'outer_r': sp['ro'], 'gap': sp['gap'], **sc})
full = pd.DataFrame(rows)
full.to_csv('/mnt/data/fullrange_core_template_full.csv', index=False)
valid = full[full['valid'] == True].copy()
mean_scores = valid.groupby(['beta', 'target'], as_index=False)[['closed', 'axial', 'square', 'cube', 'margin', 'face_mass', 'face_occ', 'inner_r', 'outer_r', 'gap']].mean(numeric_only=True)
winner_counts = valid.groupby(['beta', 'target', 'winner']).size().reset_index(name='n')
totals = valid.groupby(['beta', 'target']).size().reset_index(name='total')
winner_frac = winner_counts.merge(totals, on=['beta', 'target'])
winner_frac['frac'] = winner_frac['n'] / winner_frac['total']
pivot = winner_frac.pivot_table(index=['beta', 'target'], columns='winner', values='frac', fill_value=0).reset_index()
summary = mean_scores.merge(pivot, on=['beta', 'target'], how='left', suffixes=('_score', '_frac')).fillna(0)
summary.to_csv('/mnt/data/fullrange_core_template_summary.csv', index=False)
dom = winner_frac.sort_values(['beta', 'target', 'frac'], ascending=[True, True, False]).groupby(['beta', 'target']).head(1)
dom.to_csv('/mnt/data/fullrange_core_template_dominant.csv', index=False)

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
for ax, beta in zip(axes.ravel(), BETAS):
    sub = summary[summary['beta'] == beta].sort_values('target')
    for col, lab, style in [('closed_frac', 'closed', '-'), ('axial_frac', 'axial', '--'), ('square_frac', 'square', ':'), ('cube_frac', 'cube', '-.')]:
        ax.plot(sub['target'], sub[col], linestyle=style, label=lab)
    ax.axvspan(0.75, 0.85, color='gray', alpha=0.12)
    ax.set_title(f'β={beta:g}')
    ax.grid(True, alpha=0.25)
for ax in axes[:, 0]:
    ax.set_ylabel('winner fraction')
for ax in axes[-1]:
    ax.set_xlabel('normalized level')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('/mnt/data/fullrange_core_template_dominance.png', dpi=160, bbox_inches='tight')
plt.close(fig)
