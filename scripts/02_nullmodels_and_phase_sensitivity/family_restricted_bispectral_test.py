
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util
import re

spec = importlib.util.spec_from_file_location("pvn", "/mnt/data/phase_vs_nullmodels.py")
pvn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pvn)

GEOMETRIES = pvn.GEOMETRIES
CRITERIA = pvn.CRITERIA

# Metadata for 3x3x3 local xyz FFT bins
QMETA = {}
for i in range(3):
    for j in range(3):
        for k in range(3):
            q = (i, j, k)
            label = pvn.gpm.q_label(q)
            fam_exact = pvn.q_family(label)
            support = sum(v != 0 for v in q)
            fam_support = "const" if support == 0 else f"{support}-axis"
            QMETA[q] = {
                "label": label,
                "fam_exact": fam_exact,
                "fam_support": fam_support,
            }

def support_family_from_exact(exact_family: str) -> str:
    if exact_family == "const":
        return "const"
    n = int(exact_family.split("-")[0])
    return f"{n}-axis"

def restricted_triad_coherence(xyz, family, mode="support"):
    F = np.fft.fftn(xyz)
    shape = xyz.shape
    if mode == "support":
        eligible = [q for q, m in QMETA.items() if q != (0, 0, 0) and m["fam_support"] == family]
    elif mode == "exact":
        eligible = [q for q, m in QMETA.items() if q != (0, 0, 0) and m["fam_exact"] == family]
    else:
        raise ValueError(f"unknown mode {mode!r}")
    eligible_set = set(eligible)
    seen = set()
    Bs = []
    for q1 in eligible:
        for q2 in eligible:
            q3 = ((-(q1[0] + q2[0])) % shape[0], (-(q1[1] + q2[1])) % shape[1], (-(q1[2] + q2[2])) % shape[2])
            if q3 == (0, 0, 0) or q3 not in eligible_set:
                continue
            triple = tuple(sorted([q1, q2, q3]))
            if triple in seen:
                continue
            seen.add(triple)
            b = F[q1] * F[q2] * np.conj(F[q3])
            if np.abs(b) > 1e-12:
                Bs.append(b)
    if not Bs:
        return 0.0, 0, len(eligible)
    Bs = np.asarray(Bs)
    coh = float(np.abs(Bs.sum()) / (np.abs(Bs).sum() + 1e-12))
    return coh, len(Bs), len(eligible)

def summarize_compare(real_val, sur_vals):
    mu = float(np.mean(sur_vals))
    sd = float(np.std(sur_vals, ddof=0))
    z = float((real_val - mu) / (sd + 1e-12))
    exceed = float(np.mean(real_val > sur_vals))
    p95 = float(np.quantile(sur_vals, 0.95))
    return mu, sd, z, exceed, p95

def pick_winner(rows, criterion_key):
    col = CRITERIA[criterion_key]
    sub = [r for r in rows if r["local_mode_index"] > 1]
    return max(sub, key=lambda r: r[col])

def plot_mean_compare(summary, prefix, out_png, title):
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for crit in ["score_q", "score_iso", "q_only", "xyz_only"]:
        sub = summary[summary["criterion"] == crit].sort_values("beta")
        ax.plot(sub["beta"], sub[f"real_{prefix}_mean"], marker="o", label=f"{crit}: real")
        ax.plot(sub["beta"], sub[f"sur_{prefix}_mean"], marker="x", linestyle="--", label=f"{crit}: surrogate")
    ax.set_xlabel("beta")
    ax.set_ylabel(prefix)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def plot_exceed(summary, prefix, out_png, title):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for crit in ["score_q", "score_iso", "q_only", "xyz_only"]:
        sub = summary[summary["criterion"] == crit].sort_values("beta")
        ax.plot(sub["beta"], sub[f"{prefix}_exceed_mean"], marker="o", label=crit)
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
    parser.add_argument("--n-draws", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260320)
    parser.add_argument("--full-csv", type=str, default="/mnt/data/family_restricted_bispectral_full.csv")
    parser.add_argument("--summary-csv", type=str, default="/mnt/data/family_restricted_bispectral_summary.csv")
    parser.add_argument("--support-plot", type=str, default="/mnt/data/family_restricted_bispectral_support.png")
    parser.add_argument("--support-exceed-plot", type=str, default="/mnt/data/family_restricted_bispectral_support_exceed.png")
    parser.add_argument("--exact-exceed-plot", type=str, default="/mnt/data/family_restricted_bispectral_exact_exceed.png")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    betas = [float(x) for x in args.betas.split(",") if x.strip()]

    rows = []
    for geom, lengths in GEOMETRIES:
        for beta in betas:
            base = pvn.solve_mode_xyz(lengths, beta, pts_per_cell=args.pts_per_cell, modes=args.modes)
            for criterion in CRITERIA:
                winner = pick_winner(base, criterion)
                exact_family = winner["q_family"]
                support_family = support_family_from_exact(exact_family)

                real_exact, triads_exact, elig_exact = restricted_triad_coherence(winner["xyz"], exact_family, mode="exact")
                real_support, triads_support, elig_support = restricted_triad_coherence(winner["xyz"], support_family, mode="support")

                sur_exact = []
                sur_support = []
                for _ in range(args.n_draws):
                    s = pvn.surrogate_fft_phase(winner["xyz"], rng)
                    sur_exact.append(restricted_triad_coherence(s, exact_family, mode="exact")[0])
                    sur_support.append(restricted_triad_coherence(s, support_family, mode="support")[0])
                sur_exact = np.asarray(sur_exact)
                sur_support = np.asarray(sur_support)

                mu_e, sd_e, z_e, exc_e, p95_e = summarize_compare(real_exact, sur_exact)
                mu_s, sd_s, z_s, exc_s, p95_s = summarize_compare(real_support, sur_support)

                rows.append({
                    "geometry": geom,
                    "beta": beta,
                    "criterion": criterion,
                    "winner_mode_index": winner["local_mode_index"],
                    "winner_q": winner["dominant_q"],
                    "winner_exact_family": exact_family,
                    "winner_support_family": support_family,
                    "real_exact": real_exact,
                    "sur_exact_mean": mu_e,
                    "sur_exact_std": sd_e,
                    "z_exact": z_e,
                    "exact_exceed": exc_e,
                    "sur_exact_p95": p95_e,
                    "triads_exact": triads_exact,
                    "eligible_exact": elig_exact,
                    "real_support": real_support,
                    "sur_support_mean": mu_s,
                    "sur_support_std": sd_s,
                    "z_support": z_s,
                    "support_exceed": exc_s,
                    "sur_support_p95": p95_s,
                    "triads_support": triads_support,
                    "eligible_support": elig_support,
                })

    full = pd.DataFrame(rows)
    summary = full.groupby(["criterion", "beta"]).agg(
        real_exact_mean=("real_exact", "mean"),
        sur_exact_mean=("sur_exact_mean", "mean"),
        z_exact_mean=("z_exact", "mean"),
        exact_exceed_mean=("exact_exceed", "mean"),
        real_support_mean=("real_support", "mean"),
        sur_support_mean=("sur_support_mean", "mean"),
        z_support_mean=("z_support", "mean"),
        support_exceed_mean=("support_exceed", "mean"),
    ).reset_index()

    full.to_csv(args.full_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)
    plot_mean_compare(summary, "support", args.support_plot, "Support-family triad coherence vs fft-phase surrogates")
    plot_exceed(summary, "support", args.support_exceed_plot, "Support-family triad exceedance vs fft-phase surrogates")
    plot_exceed(summary, "exact", args.exact_exceed_plot, "Exact-family triad exceedance vs fft-phase surrogates")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
