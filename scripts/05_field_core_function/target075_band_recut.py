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


def target_band_metrics(u, target=0.75, bw=0.01, min_count=8, min_gap=0.5):
    A = np.abs(u)
    A = (A - A.min()) / (A.max() - A.min() + 1e-12)
    shp = A.shape
    coords = np.indices(shp).reshape(3, -1).T.astype(float)
    center = (np.array(shp) - 1.0) / 2.0
    vals = A.reshape(-1)
    radii = np.linalg.norm(coords - center, axis=1)

    mask = np.abs(vals - target) <= bw
    if mask.sum() < 2 * min_count:
        return None

    rr = radii[mask]
    ww = vals[mask]
    cc = coords[mask]

    c1, c2 = np.quantile(rr, [0.25, 0.75])
    for _ in range(50):
        lab = np.abs(rr - c1) <= np.abs(rr - c2)
        if lab.all() or (~lab).all():
            return None
        nc1 = rr[lab].mean()
        nc2 = rr[~lab].mean()
        if abs(nc1 - c1) + abs(nc2 - c2) < 1e-8:
            break
        c1, c2 = nc1, nc2

    if c1 <= c2:
        inner, outer = lab, ~lab
        ri, ro = c1, c2
    else:
        inner, outer = ~lab, lab
        ri, ro = c2, c1

    if inner.sum() < min_count or outer.sum() < min_count:
        return None
    gap = ro - ri
    if gap < min_gap:
        return None

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

    lower = vals < (target - bw)
    if lower.sum() >= min_count:
        lvec, _ = soft_sector_vec(coords[lower], vals[lower], center)
        carrier_vs_space_sector = float(
            (ivec @ ovec) / (np.linalg.norm(ivec) * np.linalg.norm(ovec) + 1e-12)
            - (ovec @ lvec) / (np.linalg.norm(ovec) * np.linalg.norm(lvec) + 1e-12)
        )
    else:
        carrier_vs_space_sector = np.nan

    return {
        'target': target,
        'bw': bw,
        'n_band': int(mask.sum()),
        'inner_count': int(inner.sum()),
        'outer_count': int(outer.sum()),
        'inner_r': float(ri),
        'outer_r': float(ro),
        'gap': float(gap),
        'same_sector': same_sector,
        'same_shell': same_shell,
        'deficit_fill': deficit_fill,
        'completion_gain': completion_gain,
        'carrier_vs_space_sector': carrier_vs_space_sector,
    }


def adaptive_target_metrics(u, target=0.75, bws=(0.005, 0.01, 0.02, 0.03), min_count=8, min_gap=0.5):
    for bw in bws:
        out = target_band_metrics(u, target=target, bw=bw, min_count=min_count, min_gap=min_gap)
        if out is not None:
            out['bw_selected'] = bw
            return out
    return None


def compare(real_val, sur_vals, lower_is_better=False):
    sur_vals = np.asarray([s for s in sur_vals if pd.notna(s)], dtype=float)
    if sur_vals.size == 0 or pd.isna(real_val):
        return np.nan, np.nan, np.nan, np.nan
    mu = float(np.mean(sur_vals))
    sd = float(np.std(sur_vals, ddof=0))
    if lower_is_better:
        z = float((mu - real_val) / (sd + 1e-12))
        exceed = float(np.mean(real_val < sur_vals))
    else:
        z = float((real_val - mu) / (sd + 1e-12))
        exceed = float(np.mean(real_val > sur_vals))
    return mu, sd, z, exceed


def main():
    rng = np.random.default_rng(20260321)
    full_rows = []
    summary_rows = []

    for gname, lengths in GEOMETRIES:
        for beta in BETAS:
            rows = ov.solve_mode_full(lengths, beta, ncell=3, pts_per_cell=5, modes=10)
            winners = ov.pick_winners(rows)
            for crit in CRITERIA.keys():
                w = winners[crit]
                real = adaptive_target_metrics(w['u'])
                if real is None:
                    full_rows.append({'geometry': gname, 'beta': beta, 'criterion': crit, 'kind': 'real', 'valid': False})
                    continue
                full_rows.append({'geometry': gname, 'beta': beta, 'criterion': crit, 'kind': 'real', 'valid': True, **real})

                sur_metrics = []
                for sidx in range(12):
                    us = pvn.surrogate_fft_phase(w['u'], rng)
                    sm = adaptive_target_metrics(us)
                    full_rows.append({'geometry': gname, 'beta': beta, 'criterion': crit, 'kind': f'fft_phase_{sidx}', 'valid': sm is not None, **(sm or {})})
                    if sm is not None:
                        sur_metrics.append(sm)

                for metric in [
                    'same_sector', 'same_shell', 'deficit_fill', 'completion_gain',
                    'carrier_vs_space_sector', 'n_band', 'inner_r', 'outer_r', 'gap', 'bw_selected'
                ]:
                    mu, sd, z, exc = compare(real.get(metric, np.nan), [s.get(metric, np.nan) for s in sur_metrics], lower_is_better=False)
                    summary_rows.append({
                        'geometry': gname,
                        'beta': beta,
                        'criterion': crit,
                        'metric': metric,
                        'real': real.get(metric, np.nan),
                        'fft_phase_mean': mu,
                        'fft_phase_sd': sd,
                        'z_vs_fft_phase': z,
                        'exceed_vs_fft_phase': exc,
                        'n_sur_valid': len(sur_metrics),
                    })

    full_df = pd.DataFrame(full_rows)
    summary_df = pd.DataFrame(summary_rows)
    beta_mean = summary_df.groupby(['beta', 'metric'], as_index=False)[['real', 'fft_phase_mean', 'z_vs_fft_phase', 'exceed_vs_fft_phase', 'n_sur_valid']].mean(numeric_only=True)
    bw_stats = full_df[(full_df['kind'] == 'real') & (full_df['valid'] == True)].groupby('beta', as_index=False)['bw_selected'].agg(['mean', 'min', 'max', 'count']).reset_index()

    full_df.to_csv('/mnt/data/target075_band_full.csv', index=False)
    summary_df.to_csv('/mnt/data/target075_band_summary.csv', index=False)
    beta_mean.to_csv('/mnt/data/target075_band_beta_mean.csv', index=False)
    bw_stats.to_csv('/mnt/data/target075_band_bw_stats.csv', index=False)

    for metric, title, out in [
        ('same_sector', '0.75-band inner–outer sector coupling', '/mnt/data/target075_samelevel_coupling.png'),
        ('deficit_fill', '0.75-band complementarity proxy', '/mnt/data/target075_complementarity.png'),
        ('carrier_vs_space_sector', '0.75-band carrier vs lower-space sector advantage', '/mnt/data/target075_carrier_vs_space.png'),
    ]:
        sub = beta_mean[beta_mean['metric'] == metric].sort_values('beta')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(sub['beta'], sub['real'], marker='o', label='real')
        ax.plot(sub['beta'], sub['fft_phase_mean'], marker='s', label='fft-phase')
        if metric != 'same_sector':
            ax.axhline(0.0, color='k', linestyle=':', linewidth=1)
        ax.set_title(title)
        ax.set_xlabel('beta')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out, dpi=160, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sub = beta_mean[beta_mean['metric'] == 'same_sector'].sort_values('beta')
    ax.plot(sub['beta'], sub['exceed_vs_fft_phase'], marker='o', label='same_sector exceed')
    sub2 = beta_mean[beta_mean['metric'] == 'carrier_vs_space_sector'].sort_values('beta')
    ax.plot(sub2['beta'], sub2['exceed_vs_fft_phase'], marker='s', label='carrier_vs_space exceed')
    ax.axhline(0.5, color='k', linestyle=':', linewidth=1)
    ax.set_title('0.75-band exceedance vs fft-phase')
    ax.set_xlabel('beta')
    ax.set_ylabel('exceedance')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig('/mnt/data/target075_exceedance.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
