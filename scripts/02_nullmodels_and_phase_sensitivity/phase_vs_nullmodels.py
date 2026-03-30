
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
import importlib.util
import re

spec = importlib.util.spec_from_file_location("gpm","/mnt/data/geometry_phase_map.py")
gpm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpm)

GEOMETRIES = [
    ("cubic", (1.0, 1.0, 1.0)),
    ("tet_1.05", (1.0, 1.0, 1.05)),
    ("tet_1.10", (1.0, 1.0, 1.10)),
    ("ortho_1.05_1.20", (1.0, 1.05, 1.20)),
    ("tet_1.20", (1.0, 1.0, 1.20)),
    ("ortho_1.10_1.30", (1.0, 1.10, 1.30)),
    ("tet_1.35", (1.0, 1.0, 1.35)),
    ("ortho_1.20_1.50", (1.0, 1.20, 1.50)),
    ("tet_1.50", (1.0, 1.0, 1.50)),
]

CRITERIA = {
    "score_q": "score_q",
    "score_iso": "score_iso",
    "q_only": "score_q_only",
    "xyz_only": "score_xyz_only",
}

def q_family(label):
    if label == "const":
        return "const"
    vals = []
    for part in label.split("+"):
        m = re.match(r"([XYZ])(\d+)", part)
        if m is None:
            raise ValueError(f"bad q label part: {part!r} in label {label!r}")
        vals.append(int(m.group(2)))
    vals = tuple(sorted(vals))
    return f"{len(vals)}-axis:{vals}"

def local_metrics_from_xyz(xyz):
    fft = np.fft.fftn(xyz)
    power = np.abs(fft) ** 2
    power[0, 0, 0] = 0.0
    q = np.unravel_index(np.argmax(power), power.shape)
    q_label = gpm.q_label(q)
    q_contrast = float(power[q] / (power.sum() + 1e-12))
    mean_abs_xyz = float(np.mean(np.abs(xyz)))
    return q_label, q_family(q_label), q_contrast, mean_abs_xyz

def solve_mode_xyz(lengths, beta, ncell=3, pts_per_cell=5, modes=10):
    Lx, Ly, Lz = lengths
    Nx = ncell * pts_per_cell + 1
    Ny = ncell * pts_per_cell + 1
    Nz = ncell * pts_per_cell + 1
    A, hx, hy, hz = gpm.laplacian_3d(Nx, Ny, Nz, ncell * Lx, ncell * Ly, ncell * Lz, beta)
    vals, vecs = spla.eigsh(A, k=modes, which="SM", tol=1e-6)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    rows = []
    for idx in range(modes):
        u = vecs[:, idx].reshape((Nx, Ny, Nz))
        xyz = gpm.local_xyz_array(u, ncell=ncell, pts_per_cell=pts_per_cell)
        anis = gpm.density_anisotropy(u, hx, hy, hz)
        B = gpm.boundary_ratio(u, hx, hy, hz)
        q_label, qfam, q_contrast, mean_abs_xyz = local_metrics_from_xyz(xyz)
        rows.append({
            "local_mode_index": idx + 1,
            "eigenvalue": float(vals[idx]),
            "xyz": xyz,
            "field_anisotropy": anis,
            "boundary_ratio": B,
            "dominant_q": q_label,
            "q_family": qfam,
            "q_contrast": q_contrast,
            "mean_abs_xyz": mean_abs_xyz,
            "score_q": mean_abs_xyz * q_contrast,
            "score_iso": mean_abs_xyz * anis,
            "score_q_only": q_contrast,
            "score_xyz_only": mean_abs_xyz,
        })
    return rows

def consensus_label(values):
    counts = pd.Series(values).value_counts().sort_index()
    max_count = counts.max()
    label = sorted(counts[counts == max_count].index)[0]
    return label, int(max_count)

def family_shell_groups(shape=(3,3,3)):
    cx, cy, cz = [(n - 1) // 2 for n in shape]
    groups = {}
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dx, dy, dz = abs(i - cx), abs(j - cy), abs(k - cz)
                shell = tuple(sorted((dx, dy, dz)))
                groups.setdefault(shell, []).append((i, j, k))
    return list(groups.values())

SHELL_GROUPS = family_shell_groups()

def surrogate_cell_shuffle(xyz, rng):
    flat = xyz.ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(xyz.shape)

def surrogate_shell_shuffle(xyz, rng):
    out = xyz.copy()
    for group in SHELL_GROUPS:
        vals = np.array([xyz[idx] for idx in group], dtype=float)
        rng.shuffle(vals)
        for idx, v in zip(group, vals):
            out[idx] = v
    return out

def surrogate_fft_phase(xyz, rng):
    n1, n2, n3 = xyz.shape
    F = np.fft.fftn(xyz)
    A = np.abs(F)
    G = np.zeros_like(F, dtype=np.complex128)
    visited = set()
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                idx = (i, j, k)
                if idx in visited:
                    continue
                neg = ((-i) % n1, (-j) % n2, (-k) % n3)
                if idx == neg:
                    G[idx] = A[idx] * np.sign(F[idx].real if A[idx] > 0 else 1.0)
                    visited.add(idx)
                else:
                    phi = rng.uniform(0.0, 2.0 * np.pi)
                    G[idx] = A[idx] * np.exp(1j * phi)
                    G[neg] = A[neg] * np.exp(-1j * phi)
                    visited.add(idx)
                    visited.add(neg)
    out = np.fft.ifftn(G).real
    return out

NULLS = {
    "cell_shuffle": surrogate_cell_shuffle,
    "shell_shuffle": surrogate_shell_shuffle,
    "fft_phase": surrogate_fft_phase,
}

def winner_bundle(rows):
    rows = [r for r in rows if r["local_mode_index"] > 1]
    winner_rows = []
    for cname, col in CRITERIA.items():
        best = max(rows, key=lambda r: r[col])
        winner_rows.append({
            "criterion": cname,
            "dominant_q": best["dominant_q"],
            "q_family": best["q_family"],
            "local_mode_index": best["local_mode_index"],
            "criterion_value": best[col],
            "q_contrast": best["q_contrast"],
            "mean_abs_xyz": best["mean_abs_xyz"],
            "boundary_ratio": best["boundary_ratio"],
            "field_anisotropy": best["field_anisotropy"],
        })
    wdf = pd.DataFrame(winner_rows)
    fam_cons, fam_count = consensus_label(wdf["q_family"])
    q_cons, q_count = consensus_label(wdf["dominant_q"])
    return wdf, {
        "consensus_family": fam_cons,
        "consensus_family_count": fam_count,
        "consensus_q": q_cons,
        "consensus_q_count": q_count,
        "mean_winner_q_contrast": float(wdf["q_contrast"].mean()),
        "mean_winner_xyz": float(wdf["mean_abs_xyz"].mean()),
    }

def make_surrogate_rows(base_rows, null_name, rng):
    fn = NULLS[null_name]
    rows = []
    for r in base_rows:
        xyz_s = fn(r["xyz"], rng)
        q_label, qfam, q_contrast, mean_abs_xyz = local_metrics_from_xyz(xyz_s)
        rows.append({
            "local_mode_index": r["local_mode_index"],
            "eigenvalue": r["eigenvalue"],
            "field_anisotropy": r["field_anisotropy"],
            "boundary_ratio": r["boundary_ratio"],
            "dominant_q": q_label,
            "q_family": qfam,
            "q_contrast": q_contrast,
            "mean_abs_xyz": mean_abs_xyz,
            "score_q": mean_abs_xyz * q_contrast,
            "score_iso": mean_abs_xyz * r["field_anisotropy"],
            "score_q_only": q_contrast,
            "score_xyz_only": mean_abs_xyz,
        })
    return rows

def plot_metric(summary, metric, title, out_png):
    betas = sorted(summary["beta"].unique())
    nulls = ["cell_shuffle", "shell_shuffle", "fft_phase"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for null_name in nulls:
        sub = summary[summary["null_type"] == null_name].sort_values("beta")
        ax.plot(sub["beta"], sub[metric], marker="o", label=null_name)
    ax.set_xlabel("beta")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
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
    parser.add_argument("--n-draws", type=int, default=24)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--full-csv", type=str, default="/mnt/data/phase_vs_nullmodels_full.csv")
    parser.add_argument("--summary-csv", type=str, default="/mnt/data/phase_vs_nullmodels_summary.csv")
    parser.add_argument("--real-csv", type=str, default="/mnt/data/phase_vs_nullmodels_real_baseline.csv")
    parser.add_argument("--family-plot", type=str, default="/mnt/data/phase_vs_nullmodels_family_match.png")
    parser.add_argument("--q-plot", type=str, default="/mnt/data/phase_vs_nullmodels_q_match.png")
    parser.add_argument("--consensus-plot", type=str, default="/mnt/data/phase_vs_nullmodels_consensus_strength.png")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    betas = [float(x) for x in args.betas.split(",") if x.strip()]

    baseline_rows = []
    surrogate_rows = []
    cache = {}
    for geom, lengths in GEOMETRIES:
        for beta in betas:
            base_rows = solve_mode_xyz(lengths, beta, pts_per_cell=args.pts_per_cell, modes=args.modes)
            cache[(geom, beta)] = base_rows
            real_winners, real_bundle = winner_bundle(base_rows)
            baseline_rows.append({
                "geometry": geom,
                "beta": beta,
                **real_bundle,
            })
            for null_name in NULLS:
                for draw in range(args.n_draws):
                    srows = make_surrogate_rows(base_rows, null_name, rng)
                    _, sbundle = winner_bundle(srows)
                    surrogate_rows.append({
                        "geometry": geom,
                        "beta": beta,
                        "null_type": null_name,
                        "draw": draw,
                        "real_consensus_family": real_bundle["consensus_family"],
                        "real_consensus_q": real_bundle["consensus_q"],
                        "surrogate_consensus_family": sbundle["consensus_family"],
                        "surrogate_consensus_q": sbundle["consensus_q"],
                        "family_match_real": float(sbundle["consensus_family"] == real_bundle["consensus_family"]),
                        "q_match_real": float(sbundle["consensus_q"] == real_bundle["consensus_q"]),
                        "surrogate_consensus_strength": sbundle["consensus_family_count"] / 4.0,
                        "surrogate_q_consensus_strength": sbundle["consensus_q_count"] / 4.0,
                        "real_mean_winner_q_contrast": real_bundle["mean_winner_q_contrast"],
                        "real_mean_winner_xyz": real_bundle["mean_winner_xyz"],
                        "surrogate_mean_winner_q_contrast": sbundle["mean_winner_q_contrast"],
                        "surrogate_mean_winner_xyz": sbundle["mean_winner_xyz"],
                    })

    base_df = pd.DataFrame(baseline_rows)
    full_df = pd.DataFrame(surrogate_rows)
    base_df.to_csv(args.real_csv, index=False)
    full_df.to_csv(args.full_csv, index=False)

    summary = full_df.groupby(["null_type", "beta"]).agg(
        family_match_real=("family_match_real", "mean"),
        q_match_real=("q_match_real", "mean"),
        surrogate_consensus_strength=("surrogate_consensus_strength", "mean"),
        surrogate_q_consensus_strength=("surrogate_q_consensus_strength", "mean"),
        surrogate_mean_winner_q_contrast=("surrogate_mean_winner_q_contrast", "mean"),
        surrogate_mean_winner_xyz=("surrogate_mean_winner_xyz", "mean"),
    ).reset_index()

    real_beta = base_df.groupby("beta").agg(
        real_mean_winner_q_contrast=("mean_winner_q_contrast", "mean"),
        real_mean_winner_xyz=("mean_winner_xyz", "mean"),
    ).reset_index()
    summary = summary.merge(real_beta, on="beta", how="left")
    summary.to_csv(args.summary_csv, index=False)

    plot_metric(summary, "family_match_real", "Null-model test: family match to real phase", args.family_plot)
    plot_metric(summary, "q_match_real", "Null-model test: exact q match to real phase", args.q_plot)
    plot_metric(summary, "surrogate_consensus_strength", "Null-model test: surrogate family consensus strength", args.consensus_plot)

    print("Saved:")
    print(args.real_csv)
    print(args.full_csv)
    print(args.summary_csv)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
