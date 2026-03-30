
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def robin_1d_matrix(N: int, L: float, beta: float):
    """
    Simple discrete Robin surrogate:
    path-graph Laplacian on [0,L] with boundary penalty beta/h.
    This is a stable pilot discretization for continuation studies.
    """
    h = L / (N - 1)
    main = np.full(N, 2.0 / h**2)
    main[0] = main[-1] = 1.0 / h**2 + beta / h
    off = np.full(N - 1, -1.0 / h**2)
    A = sp.diags([off, main, off], [-1, 0, 1], format="csr")
    return A, h


def laplacian_3d(Nx, Ny, Nz, Lx, Ly, Lz, beta):
    Ax, hx = robin_1d_matrix(Nx, Lx, beta)
    Ay, hy = robin_1d_matrix(Ny, Ly, beta)
    Az, hz = robin_1d_matrix(Nz, Lz, beta)

    Ix = sp.eye(Nx, format="csr")
    Iy = sp.eye(Ny, format="csr")
    Iz = sp.eye(Nz, format="csr")

    A = (
        sp.kron(sp.kron(Ax, Iy), Iz)
        + sp.kron(sp.kron(Ix, Ay), Iz)
        + sp.kron(sp.kron(Ix, Iy), Az)
    )
    return A, hx, hy, hz


def density_anisotropy(u, hx, hy, hz):
    rho = u * u
    rho /= rho.sum() * hx * hy * hz

    Nx, Ny, Nz = u.shape
    xs = np.linspace(0.0, hx * (Nx - 1), Nx)
    ys = np.linspace(0.0, hy * (Ny - 1), Ny)
    zs = np.linspace(0.0, hz * (Nz - 1), Nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    xb = (rho * X).sum() * hx * hy * hz
    yb = (rho * Y).sum() * hx * hy * hz
    zb = (rho * Z).sum() * hx * hy * hz

    dx = X - xb
    dy = Y - yb
    dz = Z - zb

    M = np.array(
        [
            [(rho * dx * dx).sum(), (rho * dx * dy).sum(), (rho * dx * dz).sum()],
            [(rho * dy * dx).sum(), (rho * dy * dy).sum(), (rho * dy * dz).sum()],
            [(rho * dz * dx).sum(), (rho * dz * dy).sum(), (rho * dz * dz).sum()],
        ]
    ) * hx * hy * hz

    evals = np.linalg.eigvalsh(M)
    evals = np.clip(evals, 1e-14, None)
    return float(evals.min() / evals.max())


def boundary_ratio(u, hx, hy, hz):
    vol = (u * u).sum() * hx * hy * hz
    B = 0.0
    B += (u[0, :, :] ** 2).sum() * hy * hz + (u[-1, :, :] ** 2).sum() * hy * hz
    B += (u[:, 0, :] ** 2).sum() * hx * hz + (u[:, -1, :] ** 2).sum() * hx * hz
    B += (u[:, :, 0] ** 2).sum() * hx * hy + (u[:, :, -1] ** 2).sum() * hx * hy
    return float(B / (vol + 1e-12))


def local_xyz_array(u, ncell, pts_per_cell):
    arr = np.zeros((ncell, ncell, ncell), dtype=float)
    offsets = [
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1),  (1, -1, 1),  (1, 1, -1),  (1, 1, 1),
    ]
    signs = np.array([a * b * c for a, b, c in offsets], dtype=float)

    for i in range(ncell):
        for j in range(ncell):
            for k in range(ncell):
                cx = i * pts_per_cell + pts_per_cell // 2
                cy = j * pts_per_cell + pts_per_cell // 2
                cz = k * pts_per_cell + pts_per_cell // 2
                vals = np.array([u[cx + a, cy + b, cz + c] for a, b, c in offsets], dtype=float)
                arr[i, j, k] = np.mean(signs * vals)
    return arr


def q_label(q):
    parts = []
    for axis, val in zip("XYZ", q):
        if val != 0:
            parts.append(f"{axis}{val}")
    return "+".join(parts) if parts else "const"


def analyze_geometry(name, cell_lengths, beta, ncell=3, pts_per_cell=5, modes=20):
    Lx, Ly, Lz = cell_lengths
    Nx = ncell * pts_per_cell + 1
    Ny = ncell * pts_per_cell + 1
    Nz = ncell * pts_per_cell + 1

    A, hx, hy, hz = laplacian_3d(
        Nx, Ny, Nz,
        ncell * Lx, ncell * Ly, ncell * Lz,
        beta,
    )
    vals, vecs = spla.eigsh(A, k=modes, which="SM", tol=1e-6)
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]

    best = None
    for m in range(1, modes):  # skip the trivial lowest mode
        u = vecs[:, m].reshape((Nx, Ny, Nz))
        xyz = local_xyz_array(u, ncell=ncell, pts_per_cell=pts_per_cell)

        fft = np.fft.fftn(xyz)
        power = np.abs(fft) ** 2
        power[0, 0, 0] = 0.0

        q = np.unravel_index(np.argmax(power), power.shape)
        q_contrast = float(power[q] / (power.sum() + 1e-12))
        mean_abs_xyz = float(np.mean(np.abs(xyz)))
        anis = density_anisotropy(u, hx, hy, hz)
        B = boundary_ratio(u, hx, hy, hz)

        score = mean_abs_xyz * q_contrast
        row = {
            "geometry": name,
            "beta": beta,
            "mode_index": m + 1,
            "eigenvalue": float(vals[m]),
            "dominant_q": q_label(q),
            "q_contrast": q_contrast,
            "mean_abs_xyz": mean_abs_xyz,
            "field_anisotropy": anis,
            "boundary_ratio": B,
            "score": float(score),
        }
        if best is None or row["score"] > best["score"]:
            best = row

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas", type=str, default="0,1,5")
    parser.add_argument("--pts-per-cell", type=int, default=5)
    parser.add_argument("--modes", type=int, default=20)
    parser.add_argument("--csv", type=str, default="")
    args = parser.parse_args()

    betas = [float(x) for x in args.betas.split(",") if x.strip()]
    geometries = [
        ("cubic", (1.0, 1.0, 1.0)),
        ("tetragonal_mild", (1.0, 1.0, 1.2)),
        ("tetragonal_strong", (1.0, 1.0, 1.5)),
        ("orthorhombic", (1.0, 1.2, 1.5)),
    ]

    rows = []
    for name, cell_lengths in geometries:
        for beta in betas:
            rows.append(
                analyze_geometry(
                    name=name,
                    cell_lengths=cell_lengths,
                    beta=beta,
                    ncell=3,
                    pts_per_cell=args.pts_per_cell,
                    modes=args.modes,
                )
            )

    df = pd.DataFrame(rows)
    if args.csv:
        df.to_csv(args.csv, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
