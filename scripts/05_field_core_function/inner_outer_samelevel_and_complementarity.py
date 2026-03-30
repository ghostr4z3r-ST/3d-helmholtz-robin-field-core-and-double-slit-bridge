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

GEOMETRIES = pvn.GEOMETRIES
CRITERIA = ov.CRITERIA
BETAS = [0.0, 1.0, 1.5, 2.0, 5.0, 10.0]

CANON = []
SHELL_TYPE = []
for i in range(3):
    for j in range(3):
        for k in range(3):
            if (i, j, k) == (1, 1, 1):
                continue
            d = np.array([i - 1, j - 1, k - 1], dtype=float)
            d /= np.linalg.norm(d)
            CANON.append(d)
            n_center = sum(v == 1 for v in (i, j, k))
            SHELL_TYPE.append({0: 'corner', 1: 'edge', 2: 'face'}[n_center])
CANON = np.array(CANON, dtype=float)
SHELL_KEYS = ['face', 'edge', 'corner']


def soft_sector_vec(coords, weights, center, kappa=8.0):
    d = coords - center
    n = np.linalg.norm(d, axis=1)
    keep = n > 1e-9
    if not np.any(keep):
        return np.zeros(len(CANON)), np.zeros(3)
    d = d[keep] / n[keep, None]
    w = weights[keep]
    sims = d @ CANON.T
    W = np.exp(kappa * sims)
    W /= W.sum(axis=1, keepdims=True)
    vec = (w[:, None] * W).sum(axis=0)
    vec /= vec.sum() + 1e-12
    shell = np.zeros(3)
    for val, st in zip(vec, SHELL_TYPE):
        shell[SHELL_KEYS.index(st)] += val
    shell /= shell.sum() + 1e-12
    return vec, shell


def band_metrics_from_u(u, levels=np.linspace(0.05, 0.95, 10), bw=0.01, min_gap=1.0, min_count=10):
    A = np.abs(u)
    A = (A - A.min()) / (A.max() - A.min() + 1e-12)
    shp = A.shape
    coords = np.indices(shp).reshape(3, -1).T.astype(float)
    center = (np.array(shp) - 1.0) / 2.0
    radii = np.linalg.norm(coords - center, axis=1)
    vals = A.reshape(-1)

    band_rows = []
    for lev in levels:
        mask = np.abs(vals - lev) <= bw
        if mask.sum() < 2 * min_count:
            continue
        rr = radii[mask]
        ww = vals[mask]
        cc = coords[mask]
        c1, c2 = np.quantile(rr, [0.25, 0.75])
        for _ in range(30):
            lab = np.abs(rr - c1) <= np.abs(rr - c2)
            if lab.all() or (~lab).all():
                break
            nc1 = rr[lab].mean()
            nc2 = rr[~lab].mean()
            if abs(nc1 - c1) + abs(nc2 - c2) < 1e-7:
                break
            c1, c2 = nc1, nc2
        if c1 <= c2:
            inner, outer = lab, ~lab
            ri, ro = c1, c2
        else:
            inner, outer = ~lab, lab
            ri, ro = c2, c1
        if inner.sum() < min_count or outer.sum() < min_count:
            continue
        gap = ro - ri
        if gap < min_gap:
            continue

        ivec, ishell = soft_sector_vec(cc[inner], ww[inner], center)
        ovec, oshell = soft_sector_vec(cc[outer], ww[outer], center)
        same_sector = float(ivec @ ovec / (np.linalg.norm(ivec) * np.linalg.norm(ovec) + 1e-12))
        same_shell = float(ishell @ oshell / (np.linalg.norm(ishell) * np.linalg.norm(oshell) + 1e-12))

        q25, q75 = np.quantile(ivec, [0.25, 0.75])
        weak = ivec <= q25
        strong = ivec >= q75
        deficit_fill = float(ovec[weak].mean() - ovec[strong].mean())
        comb = ivec + ovec
        comb /= comb.sum() + 1e-12
        completion_gain = float(comb.min() - ivec.min())

        band_rows.append({
            'level': float(lev),
            'gap': float(gap),
            'inner_r': float(ri),
            'outer_r': float(ro),
            'inner_count': int(inner.sum()),
            'outer_count': int(outer.sum()),
            'same_sector': same_sector,
            'same_shell': same_shell,
            'deficit_fill': deficit_fill,
            'completion_gain': completion_gain,
        })

    if not band_rows:
        return None, pd.DataFrame()
    bands = pd.DataFrame(band_rows)
    outer = bands.iloc[bands['outer_r'].argmax()]
    out = {
        'n_valid_bands': int(len(bands)),
        'same_sector_mean': float(bands['same_sector'].mean()),
        'same_shell_mean': float(bands['same_shell'].mean()),
        'deficit_fill_mean': float(bands['deficit_fill'].mean()),
        'completion_gain_mean': float(bands['completion_gain'].mean()),
        'same_sector_outer': float(outer['same_sector']),
        'same_shell_outer': float(outer['same_shell']),
        'deficit_fill_outer': float(outer['deficit_fill']),
        'completion_gain_outer': float(outer['completion_gain']),
        'outer_level': float(outer['level']),
        'outer_radius': float(outer['outer_r']),
        'outer_gap': float(outer['gap']),
    }
    return out, bands


def compare(real_val, sur_vals, lower_is_better=False):
    sur_vals = np.asarray(sur_vals, dtype=float)
    mu = float(np.nanmean(sur_vals))
    sd = float(np.nanstd(sur_vals, ddof=0))
    if lower_is_better:
        z = float((mu - real_val) / (sd + 1e-12))
        exc = float(np.mean(real_val < sur_vals))
    else:
        z = float((real_val - mu) / (sd + 1e-12))
        exc = float(np.mean(real_val > sur_vals))
    return mu, sd, z, exc


def main():
    rng = np.random.default_rng(12345)
    full_rows = []
    band_rows = []
    summary_rows = []
    n_sur = 8

    for gname, lengths in GEOMETRIES:
        for beta in BETAS:
            rows = ov.solve_mode_full(lengths, beta, ncell=3, pts_per_cell=5, modes=10)
            winners = ov.pick_winners(rows)
            for crit in CRITERIA.keys():
                w = winners[crit]
                real, bands = band_metrics_from_u(w['u'])
                if real is None:
                    continue
                for _, br in bands.iterrows():
                    band_rows.append({
                        'geometry': gname, 'beta': beta, 'criterion': crit, 'kind': 'real', **br.to_dict()
                    })
                full = {
                    'geometry': gname, 'beta': beta, 'criterion': crit,
                    'q_family': w['q_family'], 'dominant_q': w['dominant_q'], **real
                }
                full_rows.append(full)

                sur_metrics = []
                for sidx in range(n_sur):
                    us = pvn.surrogate_fft_phase(w['u'], rng)
                    sm, sbands = band_metrics_from_u(us)
                    if sm is None:
                        continue
                    sur_metrics.append(sm)
                    for _, br in sbands.iterrows():
                        band_rows.append({
                            'geometry': gname, 'beta': beta, 'criterion': crit, 'kind': f'fft_phase_{sidx}', **br.to_dict()
                        })
                if not sur_metrics:
                    continue
                for metric in [
                    'n_valid_bands', 'same_sector_mean', 'same_shell_mean',
                    'deficit_fill_mean', 'completion_gain_mean',
                    'same_sector_outer', 'same_shell_outer',
                    'deficit_fill_outer', 'completion_gain_outer',
                    'outer_radius', 'outer_gap'
                ]:
                    mu, sd, z, exc = compare(real[metric], [s[metric] for s in sur_metrics], lower_is_better=False)
                    summary_rows.append({
                        'geometry': gname, 'beta': beta, 'criterion': crit, 'metric': metric,
                        'real': real[metric], 'fft_phase_mean': mu, 'fft_phase_sd': sd,
                        'z_vs_fft_phase': z, 'exceed_vs_fft_phase': exc,
                    })

    full_df = pd.DataFrame(full_rows)
    band_df = pd.DataFrame(band_rows)
    summary_df = pd.DataFrame(summary_rows)
    beta_mean = summary_df.groupby(['beta', 'metric'], as_index=False)[['real', 'fft_phase_mean', 'z_vs_fft_phase', 'exceed_vs_fft_phase']].mean(numeric_only=True)

    full_df.to_csv('/mnt/data/inner_outer_samelevel_full.csv', index=False)
    band_df.to_csv('/mnt/data/inner_outer_samelevel_bands.csv', index=False)
    summary_df.to_csv('/mnt/data/inner_outer_samelevel_summary.csv', index=False)
    beta_mean.to_csv('/mnt/data/inner_outer_samelevel_beta_mean.csv', index=False)

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for metric, ax, title in [
        ('same_sector_outer', axes[0], 'Inner–outer same-level coupling (outermost valid band)'),
        ('same_sector_mean', axes[1], 'Inner–outer same-level coupling (mean over valid bands)')
    ]:
        sub = beta_mean[beta_mean['metric'] == metric].sort_values('beta')
        ax.plot(sub['beta'], sub['real'], marker='o', label='real')
        ax.plot(sub['beta'], sub['fft_phase_mean'], marker='s', label='fft-phase')
        ax.set_title(title)
        ax.set_xlabel('beta')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('metric value')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig('/mnt/data/inner_outer_samelevel_coupling.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for metric, ax, title in [
        ('deficit_fill_outer', axes[0], 'Complementarity proxy: outer hits weak inner sectors'),
        ('completion_gain_outer', axes[1], 'Complementarity proxy: completion gain (outermost band)')
    ]:
        sub = beta_mean[beta_mean['metric'] == metric].sort_values('beta')
        ax.plot(sub['beta'], sub['real'], marker='o', label='real')
        ax.plot(sub['beta'], sub['fft_phase_mean'], marker='s', label='fft-phase')
        ax.axhline(0.0, color='k', linestyle=':', linewidth=1)
        ax.set_title(title)
        ax.set_xlabel('beta')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('metric value')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig('/mnt/data/inner_outer_complementarity.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
