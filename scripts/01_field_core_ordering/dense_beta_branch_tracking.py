
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla
from scipy.optimize import linear_sum_assignment
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

def solve_modes(lengths, beta, ncell=3, pts_per_cell=5, modes=20):
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
        fft = np.fft.fftn(xyz)
        power = np.abs(fft) ** 2
        power[0, 0, 0] = 0.0
        q = np.unravel_index(np.argmax(power), power.shape)
        q_contrast = float(power[q] / (power.sum() + 1e-12))
        mean_abs_xyz = float(np.mean(np.abs(xyz)))
        anis = gpm.density_anisotropy(u, hx, hy, hz)
        B = gpm.boundary_ratio(u, hx, hy, hz)
        rows.append(
            {
                "local_mode_index": idx + 1,
                "eigenvalue": float(vals[idx]),
                "dominant_q": gpm.q_label(q),
                "q_contrast": q_contrast,
                "mean_abs_xyz": mean_abs_xyz,
                "field_anisotropy": anis,
                "boundary_ratio": B,
                "score_q": mean_abs_xyz * q_contrast,
                "score_iso": mean_abs_xyz * anis,
                "score_q_only": q_contrast,
                "score_xyz_only": mean_abs_xyz,
            }
        )
    return vals, vecs, pd.DataFrame(rows)

def q_family(label):
    if label == "const":
        return "const"
    parts = label.split("+")
    vals = []
    for p in parts:
        m = re.match(r"([XYZ])(\d+)", p)
        vals.append(int(m.group(2)))
    vals = tuple(sorted(vals))
    n = len(parts)
    return f"{n}-axis:{vals}"

def track_geometry(name, lengths, betas, ncell=3, pts_per_cell=5, modes=20):
    all_rows = []
    prev_vecs = None
    prev_branch_ids = None
    for beta in betas:
        vals, vecs, df = solve_modes(lengths, beta, ncell=ncell, pts_per_cell=pts_per_cell, modes=modes)
        if prev_vecs is None:
            branch_ids = np.arange(1, modes + 1)
            overlaps = np.ones(modes)
            parent_branch = np.arange(1, modes + 1)
        else:
            ov = np.abs(prev_vecs.T @ vecs)
            r, c = linear_sum_assignment(-ov)
            branch_ids = np.empty(modes, dtype=int)
            overlaps = np.empty(modes, dtype=float)
            parent_branch = np.empty(modes, dtype=int)
            for rr, cc in zip(r, c):
                branch_ids[cc] = prev_branch_ids[rr]
                overlaps[cc] = ov[rr, cc]
                parent_branch[cc] = prev_branch_ids[rr]

        df.insert(0, "branch_overlap_prev", overlaps)
        df.insert(0, "parent_branch_id", parent_branch)
        df.insert(0, "branch_id", branch_ids)
        df.insert(0, "beta", beta)
        df.insert(0, "geometry", name)
        all_rows.append(df)
        prev_vecs = vecs
        prev_branch_ids = branch_ids

    return pd.concat(all_rows, ignore_index=True)

def plot_consensus(summary_df, out_png):
    geoms = list(summary_df["geometry"].drop_duplicates())
    betas = list(summary_df["beta"].drop_duplicates())
    family_to_code = {fam: i for i, fam in enumerate(sorted(summary_df["consensus_family"].dropna().unique()))}
    M = np.full((len(geoms), len(betas)), np.nan)
    for i, g in enumerate(geoms):
        for j, b in enumerate(betas):
            sub = summary_df[(summary_df.geometry == g) & (summary_df.beta == b)]
            if len(sub):
                M[i, j] = family_to_code[sub.iloc[0]["consensus_family"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(M, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([str(b) for b in betas])
    ax.set_yticks(range(len(geoms)))
    ax.set_yticklabels(geoms)
    ax.set_xlabel("beta")
    ax.set_title("Dense beta branch-tracked consensus families")

    inv = {v: k for k, v in family_to_code.items()}
    for i in range(len(geoms)):
        for j in range(len(betas)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, inv[int(M[i, j])], ha="center", va="center", color="white", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_ticks(list(inv.keys()))
    cbar.set_ticklabels([inv[k] for k in inv.keys()])
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", type=str, default="0,0.25,0.5,0.75,1,1.5,2,3,4,5,7,10")
    parser.add_argument("--pts-per-cell", type=int, default=5)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--full-csv", type=str, default="/mnt/data/dense_beta_branch_tracking_full.csv")
    parser.add_argument("--winners-csv", type=str, default="/mnt/data/dense_beta_branch_tracking_winners.csv")
    parser.add_argument("--summary-csv", type=str, default="/mnt/data/dense_beta_branch_tracking_summary.csv")
    parser.add_argument("--branch-summary-csv", type=str, default="/mnt/data/dense_beta_branch_tracking_branch_summary.csv")
    parser.add_argument("--plot", type=str, default="/mnt/data/dense_beta_branch_tracking_consensus.png")
    args = parser.parse_args()

    betas = [float(x) for x in args.betas.split(",") if x.strip()]

    frames = []
    for name, lengths in GEOMETRIES:
        frames.append(track_geometry(name, lengths, betas, pts_per_cell=args.pts_per_cell, modes=args.modes))
    full = pd.concat(frames, ignore_index=True)
    full.to_csv(args.full_csv, index=False)

    work = full[full.local_mode_index > 1].copy()
    winner_rows = []
    for (geom, beta), grp in work.groupby(["geometry", "beta"]):
        for cname, col in CRITERIA.items():
            best = grp.sort_values(col, ascending=False).iloc[0]
            winner_rows.append(
                {
                    "geometry": geom,
                    "beta": beta,
                    "criterion": cname,
                    "branch_id": int(best.branch_id),
                    "local_mode_index": int(best.local_mode_index),
                    "eigenvalue": best.eigenvalue,
                    "dominant_q": best.dominant_q,
                    "q_family": q_family(best.dominant_q),
                    "q_contrast": best.q_contrast,
                    "mean_abs_xyz": best.mean_abs_xyz,
                    "field_anisotropy": best.field_anisotropy,
                    "boundary_ratio": best.boundary_ratio,
                    "criterion_value": best[col],
                    "branch_overlap_prev": best.branch_overlap_prev,
                }
            )
    winners = pd.DataFrame(winner_rows)
    winners.to_csv(args.winners_csv, index=False)

    cons = winners.groupby(["geometry", "beta", "q_family"]).size().reset_index(name="count")
    cons = cons.sort_values(["geometry", "beta", "count", "q_family"], ascending=[True, True, False, True])
    cons_top = cons.groupby(["geometry", "beta"]).head(1).copy()
    cons_top.rename(columns={"q_family": "consensus_family", "count": "consensus_count"}, inplace=True)

    exact_cons = winners.groupby(["geometry", "beta", "dominant_q"]).size().reset_index(name="count")
    exact_cons = exact_cons.sort_values(["geometry", "beta", "count", "dominant_q"], ascending=[True, True, False, True]).groupby(["geometry", "beta"]).head(1)
    exact_cons.rename(columns={"dominant_q": "consensus_q", "count": "consensus_q_count"}, inplace=True)

    summary = winners.merge(cons_top[["geometry", "beta", "consensus_family", "consensus_count"]], on=["geometry", "beta"], how="left")
    summary = summary.merge(exact_cons[["geometry", "beta", "consensus_q", "consensus_q_count"]], on=["geometry", "beta"], how="left")
    summary = summary.groupby(["geometry", "beta"]).agg(
        consensus_family=("consensus_family", "first"),
        consensus_count=("consensus_count", "first"),
        consensus_q=("consensus_q", "first"),
        consensus_q_count=("consensus_q_count", "first"),
        mean_boundary=("boundary_ratio", "mean"),
        mean_q=("q_contrast", "mean"),
        mean_xyz=("mean_abs_xyz", "mean"),
    ).reset_index()
    summary.to_csv(args.summary_csv, index=False)

    branch_rows = []
    winners_sorted = winners.sort_values(["geometry", "criterion", "beta"])
    for (g, c), grp in winners_sorted.groupby(["geometry", "criterion"]):
        branches = grp["branch_id"].tolist()
        bets = grp["beta"].tolist()
        dom = grp["dominant_q"].tolist()
        switches = 0
        first_switch_beta = np.nan
        for i in range(1, len(branches)):
            if branches[i] != branches[i - 1]:
                switches += 1
                if np.isnan(first_switch_beta):
                    first_switch_beta = bets[i]
        branch_rows.append(
            {
                "geometry": g,
                "criterion": c,
                "winner_branch_switches": switches,
                "first_switch_beta": first_switch_beta,
            }
        )
    branch_summary = pd.DataFrame(branch_rows)
    branch_summary.to_csv(args.branch_summary_csv, index=False)

    plot_consensus(summary, args.plot)

if __name__ == "__main__":
    main()
