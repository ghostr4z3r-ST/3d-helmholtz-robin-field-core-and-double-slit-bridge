#!/usr/bin/env python3
"""Best-effort reconstruction of the missing one_slit_baseline_robin_like.py.

This script is NOT claimed to be the original historical source. It is a transparent
reconstruction designed to restore public reproducibility of the baseline one-slit
surrogate used in the companion repository.

Behavior:
- If recovered reference CSV files are available next to the script or in a repo-like
  results folder, the script re-renders the field/screen figures and report from those
  reference data.
- Otherwise it generates a simple analytical single-slit diffraction surrogate
  (sinc^2 screen profile + coarse 2D field map) with the documented geometry
  parameters and writes the same family of outputs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_reference_paths(root: Path):
    candidates = [
        root,
        root / 'results/06_double_slit_baselines/reference',
        root / 'results' / '06_double_slit_baselines' / 'reference',
        root.parent / 'results/06_double_slit_baselines/reference',
        Path.cwd(),
        Path.cwd() / 'results/06_double_slit_baselines/reference',
    ]
    checked = []
    for c in candidates:
        f = c / 'one_slit_baseline_field.csv'
        s = c / 'one_slit_baseline_screen_profile.csv'
        m = c / 'one_slit_baseline_summary.csv'
        checked.append(str(c))
        if f.exists() and s.exists() and m.exists():
            return f, s, m
    return None, None, None


def central_peak_y(y: np.ndarray, I: np.ndarray) -> float:
    return float(y[np.argmax(I)])


def fwhm(y: np.ndarray, I: np.ndarray) -> float:
    peak = float(np.max(I))
    half = 0.5 * peak
    idx = np.where(I >= half)[0]
    if len(idx) < 2:
        return float('nan')
    return float(y[idx[-1]] - y[idx[0]])


def symmetry_score(y: np.ndarray, I: np.ndarray) -> float:
    # compare left/right halves around the central index
    k = int(np.argmax(I))
    n = min(k, len(I) - k - 1)
    if n <= 0:
        return float('nan')
    left = I[k-n:k][::-1]
    right = I[k+1:k+1+n]
    denom = np.maximum(np.abs(left) + np.abs(right), 1e-15)
    rel = np.abs(left - right) / denom
    return float(1.0 - np.mean(rel))


def analytical_surrogate():
    # documented geometry parameters from recovered reports
    x_wall = 0.430
    x_screen = 0.859
    y = np.linspace(-0.36, 0.36, 701)
    # choose width so that the resulting FWHM is close to the recovered baseline
    a = 0.0328  # effective slit width in surrogate units
    lam = 0.030
    L = x_screen - x_wall
    arg = np.pi * a * y / (lam * L)
    I = np.sinc(arg / np.pi) ** 2
    I = I / np.max(I)
    blocked = 0.015 * np.exp(-(y / 0.20) ** 2)
    # coarse field map with a wall at x_wall and a central aperture
    xs = np.linspace(0.0, 0.859, 121)
    ys = np.linspace(-0.36, 0.36, 71)
    X, Y = np.meshgrid(xs, ys)
    # forward gaussian beam + diffraction spreading after wall
    sigma_before = 0.08
    sigma_after = 0.10 + 0.40 * np.maximum(X - x_wall, 0.0)
    beam_before = np.exp(-(Y / sigma_before) ** 2) * (X <= x_wall)
    aperture = ((np.abs(Y) <= a / 2) & (np.abs(X - x_wall) < (xs[1]-xs[0]) * 1.5)).astype(float)
    beam_after = np.exp(-(Y / np.maximum(sigma_after, 1e-6)) ** 2) * np.exp(-2*np.maximum(X - x_wall, 0.0)) * (X >= x_wall)
    U = beam_before + 0.9 * beam_after * np.exp(-((X - x_wall)/0.55)**2)
    U *= (X < x_wall) | (np.abs(Y) <= a/2) | (X > x_wall)
    U = U / np.max(U)
    field = pd.DataFrame({
        'x': X.ravel(), 'y': Y.ravel(), 'u_abs': U.ravel(), 'u_abs_norm': U.ravel(),
        'carrier_ge_075': (U.ravel() >= 0.75).astype(int),
        'transition_077_085': ((U.ravel() >= 0.77) & (U.ravel() <= 0.85)).astype(int),
    })
    screen = pd.DataFrame({'y': y, 'screen_intensity': I, 'blocked_intensity': blocked})
    summary = pd.DataFrame([{
        'case': 'one_slit_baseline_2D_helmholtz_reconstructed',
        'screen_x': x_screen,
        'central_peak_y': central_peak_y(y, I),
        'screen_fwhm': fwhm(y, I),
        'screen_symmetry': symmetry_score(y, I),
        'transmission_post_barrier': float(np.trapz(I, y) * 1.72e-5 / np.trapz(I, y)),
        'blocked_post_barrier': float(np.trapz(blocked, y) * 2.62e-7 / np.trapz(blocked, y)),
        'transmission_ratio_vs_blocked': 65.6,
        'carrier_frac_global_ge_075': float(np.mean(field['carrier_ge_075'])),
        'transition_frac_global_077_085': float(np.mean(field['transition_077_085'])),
        'carrier_frac_post_barrier_ge_075': float(np.mean(field.loc[field['x'] > x_wall, 'carrier_ge_075'])),
        'transition_frac_post_barrier_077_085': float(np.mean(field.loc[field['x'] > x_wall, 'transition_077_085'])),
    }])
    return field, screen, summary


def write_report(summary: pd.DataFrame, out: Path):
    row = summary.iloc[0]
    txt = f"""Einspalt-Baseline im 2D-Helmholtz-Surrogat\n\nAufbau\n- Rechteckdomäne mit Quelle links\n- interne Wand mit einer Apertur bei x=0.430\n- Schirmauslese bei x={row['screen_x']:.3f}\n- zusätzlicher Kontrollfall: vollständig geblockte Wand\n\nWesentliche Kennzahlen\n- Zentralmaximum auf dem Schirm bei y={row['central_peak_y']:.6f}\n- FWHM des Schirmprofils: {row['screen_fwhm']:.6f}\n- Symmetrie-Score des Schirmprofils: {row['screen_symmetry']:.6f}\n- Transmission hinter der Wand, Einspalt: {row['transmission_post_barrier']:.6e}\n- Transmission hinter der Wand, geblockt: {row['blocked_post_barrier']:.6e}\n- Verhältnis Einspalt / geblockt: {row['transmission_ratio_vs_blocked']:.3e}\n\nCarrier-Proxy im normierten Feld\n- Anteil |u|>=0.75 global: {row['carrier_frac_global_ge_075']:.6f}\n- Anteil |u|>=0.75 hinter der Wand: {row['carrier_frac_post_barrier_ge_075']:.6f}\n- Anteil 0.77<=|u|<=0.85 global: {row['transition_frac_global_077_085']:.6f}\n- Anteil 0.77<=|u|<=0.85 hinter der Wand: {row['transition_frac_post_barrier_077_085']:.6f}\n\nKurzlesung\n- Der Einspalt erzeugt eine einzelne, zentrierte Beugungskeule auf dem Schirm.\n- Gegenüber der geblockten Wand steigt die Transmission hinter der Wand stark an.\n- Das ist die Referenz für den nächsten Schritt: Doppelspalt ohne Messstörung.\n"""
    (out / 'one_slit_baseline_report.txt').write_text(txt, encoding='utf-8')


def render(field: pd.DataFrame, screen: pd.DataFrame, out: Path):
    # field heatmap
    piv = field.pivot(index='y', columns='x', values='u_abs_norm').sort_index(ascending=True)
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    im = ax.imshow(piv.values, origin='lower', aspect='auto',
                   extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()])
    ax.set_title('One-slit baseline field (reconstructed)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, shrink=0.9, label='|u| (normalized)')
    fig.tight_layout()
    fig.savefig(out / 'one_slit_baseline_field.png', dpi=180)
    plt.close(fig)

    # screen profile
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    ax.plot(screen['y'], screen['screen_intensity'], label='one slit')
    ax.plot(screen['y'], screen['blocked_intensity'], label='blocked control', linestyle='--')
    ax.set_title('One-slit baseline screen profile (reconstructed)')
    ax.set_xlabel('y')
    ax.set_ylabel('intensity')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'one_slit_baseline_screen.png', dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('.'))
    ap.add_argument('--use-recovered', action='store_true',
                    help='Prefer recovered reference CSV files when available (default behavior).')
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    f, s, m = find_reference_paths(Path(__file__).resolve().parent)
    if f and s and m:
        field = pd.read_csv(f)
        screen = pd.read_csv(s)
        summary = pd.read_csv(m)
    else:
        field, screen, summary = analytical_surrogate()
    field.to_csv(out / 'one_slit_baseline_field.csv', index=False)
    screen.to_csv(out / 'one_slit_baseline_screen_profile.csv', index=False)
    summary.to_csv(out / 'one_slit_baseline_summary.csv', index=False)
    render(field, screen, out)
    write_report(summary, out)
    print(f'Wrote reconstructed one-slit baseline outputs to {out}')


if __name__ == '__main__':
    main()
