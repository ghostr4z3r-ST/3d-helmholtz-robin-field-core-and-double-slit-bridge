
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util

spec = importlib.util.spec_from_file_location("pvn", "/mnt/data/phase_vs_nullmodels.py")
pvn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pvn)

GEOMETRIES = pvn.GEOMETRIES
CRITERIA = {
    "score_q": "score_q",
    "score_iso": "score_iso",
    "q_only": "score_q_only",
    "xyz_only": "score_xyz_only",
}

def unique_triads(shape):
    n1, n2, n3 = shape
    idxs = [(i, j, k) for i in range(n1) for j in range(n2) for k in range(n3)]
    idxs_nz = [q for q in idxs if q != (0, 0, 0)]
    seen = set()
    out = []
    for q1 in idxs_nz:
        for q2 in idxs_nz:
            q3 = ((-(q1[0] + q2[0])) % n1, (-(q1[1] + q2[1])) % n2, (-(q1[2] + q2[2])) % n3)
            if q3 == (0, 0, 0):
                continue
            triple = tuple(sorted([q1, q2, q3]))
            if triple in seen:
                continue
            seen.add(triple)
            out.append((q1, q2, q3))
    return out

TRIADS_3 = unique_triads((3, 3, 3))

def triad_coherence_all(xyz):
    F = np.fft.fftn(xyz)
    Bs = []
    for q1, q2, q3 in TRIADS_3:
        b = F[q1] * F[q2] * np.conj(F[q3])
        if np.abs(b) > 1e-12:
            Bs.append(b)
    if not Bs:
        return 0.0
    Bs = np.asarray(Bs)
    return float(np.abs(Bs.sum()) / (np.abs(Bs).sum() + 1e-12))

def triad_coherence_topk(xyz, k=8):
    F = np.fft.fftn(xyz)
    P = np.abs(F) ** 2
    P[0, 0, 0] = 0.0
    order = np.argsort(P.ravel())[::-1]
    top = []
    shape = xyz.shape
    for flat in order:
        q = np.unravel_index(flat, shape)
        if P[q] <= 1e-12:
            break
        top.append(q)
        if len(top) >= k:
            break
    top_set = set(top)
    seen = set()
    Bs = []
    for q1 in top:
        for q2 in top:
            q3 = ((-(q1[0] + q2[0])) % shape[0], (-(q1[1] + q2[1])) % shape[1], (-(q1[2] + q2[2])) % shape[2])
            if q3 == (0, 0, 0) or q3 not in top_set:
                continue
            triple = tuple(sorted([q1, q2, q3]))
            if triple in seen:
                continue
            seen.add(triple)
            b = F[q1] * F[q2] * np.conj(F[q3])
            if np.abs(b) > 1e-12:
                Bs.append(b)
    if not Bs:
        return 0.0
    Bs = np.asarray(Bs)
    return float(np.abs(Bs.sum()) / (np.abs(Bs).sum() + 1e-12))

def compute_mode_metrics(rows):
    out = []
    for r in rows:
        rr = dict(r)
        xyz = r["xyz"]
        rr["triad_all"] = triad_coherence_all(xyz)
        rr["triad_topk"] = triad_coherence_topk(xyz, k=8)
        out.append(rr)
    return out

def pick_winner(rows, criterion_key):
    col = CRITERIA[criterion_key]
    sub = [r for r in rows if r["local_mode_index"] > 1]
    return max(sub, key=lambda r: r[col])

def make_surrogate_metrics(xyz, rng, n_draws):
    vals_all = []
    vals_topk = []
    for _ in range(n_draws):
        s = pvn.surrogate_fft_phase(xyz, rng)
        vals_all.append(triad_coherence_all(s))
        vals_topk.append(triad_coherence_topk(s, k=8))
    return np.asarray(vals_all), np.asarray(vals_topk)

def summarize_compare(real_val, sur_vals):
    mu = float(np.mean(sur_vals))
    sd = float(np.std(sur_vals, ddof=0))
    z = float((real_val - mu) / (sd + 1e-12))
    exceed = float(np.mean(real_val > sur_vals))
    return mu, sd, z, exceed, float(np.quantile(sur_vals, 0.95))

def plot_mean_compare(summary, metric_prefix, out_png, title):
    betas = sorted(summary["beta"].unique())
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    crits = ["score_q", "score_iso", "q_only", "xyz_only"]
    for crit in crits:
        sub = summary[summary["criterion"] == crit].sort_values("beta")
        ax.plot(sub["beta"], sub[f"real_{metric_prefix}_mean"], marker="o", label=f"{crit}: real")
        ax.plot(sub["beta"], sub[f"sur_{metric_prefix}_mean"], marker="x", linestyle="--", label=f"{crit}: surrogate")
    ax.set_xlabel("beta")
    ax.set_ylabel(metric_prefix)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def plot_exceed(summary, metric_prefix, out_png, title):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    crits = ["score_q", "score_iso", "q_only", "xyz_only"]
    for crit in crits:
        sub = summary[summary["criterion"] == crit].sort_values("beta")
        ax.plot(sub["beta"], sub[f"{metric_prefix}_exceed_mean"], marker="o", label=crit)
    ax.axhline(0.5, color="k", linestyle=":", linewidth=1)
    ax.set_xlabel("beta")
    ax.set_ylabel("P(real > fft-phase surrogate)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", type=str, default="0,1,1.5,2,5,10")
    parser.add_argument("--pts-per-cell", type=int, default=5)
    parser.add_argument("--modes", type=int, default=10)
    parser.add_argument("--n-draws", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--full-csv", type=str, default="/mnt/data/bispectral_phase_lock_full.csv")
    parser.add_argument("--summary-csv", type=str, default="/mnt/data/bispectral_phase_lock_summary.csv")
    parser.add_argument("--topk-plot", type=str, default="/mnt/data/bispectral_phase_lock_topk.png")
    parser.add_argument("--all-plot", type=str, default="/mnt/data/bispectral_phase_lock_all.png")
    parser.add_argument("--exceed-topk-plot", type=str, default="/mnt/data/bispectral_phase_lock_exceed_topk.png")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    betas = [float(x) for x in args.betas.split(",") if x.strip()]

    rows = []
    for geom, lengths in GEOMETRIES:
        for beta in betas:
            base = pvn.solve_mode_xyz(lengths, beta, pts_per_cell=args.pts_per_cell, modes=args.modes)
            base = compute_mode_metrics(base)
            for criterion in CRITERIA:
                winner = pick_winner(base, criterion)
                sur_all, sur_topk = make_surrogate_metrics(winner["xyz"], rng, args.n_draws)
                mu_all, sd_all, z_all, exc_all, p95_all = summarize_compare(winner["triad_all"], sur_all)
                mu_top, sd_top, z_top, exc_top, p95_top = summarize_compare(winner["triad_topk"], sur_topk)
                rows.append({
                    "geometry": geom,
                    "beta": beta,
                    "criterion": criterion,
                    "winner_mode_index": winner["local_mode_index"],
                    "winner_q": winner["dominant_q"],
                    "winner_family": winner["q_family"],
                    "winner_q_contrast": winner["q_contrast"],
                    "winner_xyz": winner["mean_abs_xyz"],
                    "real_triad_all": winner["triad_all"],
                    "sur_triad_all_mean": mu_all,
                    "sur_triad_all_std": sd_all,
                    "triad_all_z": z_all,
                    "triad_all_exceed": exc_all,
                    "sur_triad_all_p95": p95_all,
                    "real_triad_topk": winner["triad_topk"],
                    "sur_triad_topk_mean": mu_top,
                    "sur_triad_topk_std": sd_top,
                    "triad_topk_z": z_top,
                    "triad_topk_exceed": exc_top,
                    "sur_triad_topk_p95": p95_top,
                })

    full = pd.DataFrame(rows)
    summary = full.groupby(["criterion", "beta"]).agg(
        real_all_mean=("real_triad_all", "mean"),
        sur_all_mean=("sur_triad_all_mean", "mean"),
        all_z_mean=("triad_all_z", "mean"),
        all_exceed_mean=("triad_all_exceed", "mean"),
        real_topk_mean=("real_triad_topk", "mean"),
        sur_topk_mean=("sur_triad_topk_mean", "mean"),
        topk_z_mean=("triad_topk_z", "mean"),
        topk_exceed_mean=("triad_topk_exceed", "mean"),
    ).reset_index()

    full.to_csv(args.full_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    plot_mean_compare(summary, "topk", args.topk_plot, "Top-k triad coherence vs fft-phase surrogates")
    plot_mean_compare(summary, "all", args.all_plot, "All-triad coherence vs fft-phase surrogates")
    plot_exceed(summary, "topk", args.exceed_topk_plot, "Top-k triad coherence exceedance vs fft-phase surrogates")

    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
