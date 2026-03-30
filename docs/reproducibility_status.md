# Reproducibility status

## Scope
This repository assumes the validated low-mode field framework established in:

**Reproducible Low-Mode Ordering in a 3D Helmholtz–Robin Eigenproblem on Cube-Like and Lattice-Linked Geometries**

It does not repeat the foundational artifact-hardening argument from scratch.

## Current public-core state
- **179** tracked public-core artifacts
- **177** recovered historical artifacts currently present
- **2** reconstructed public-core artifacts included to close the manuscript reproduction path

## Present recovered highlights
- Dense beta tracking and field-core robustness outputs are present.
- Opposite / carrier / corner-edge-face diagnostics are effectively complete.
- Reduction-to-minimal-core and field-core-function blocks are present in strong form.
- Double-slit, detector matrix, and effective detector law blocks are present in strong form.
- Microscopic Robin-noise support is present with summary, curves, coefficients, and report note.

## Public-core reconstructions
- `scripts/06_double_slit_baselines/one_slit_baseline_robin_like.py` is included as a transparent best-effort reconstruction from recovered baseline outputs.
- `figures/04_reduction_to_minimal_core/stepE_kernel_plus_bridge_accuracy.png` was regenerated from surviving reduction outputs and recovered bridge-geometry logic.

## Not counted as a public-core gap
- `build_atom_H.py` is treated as historical builder/provenance material rather than as a required companion-repository artifact.

## Technical note
Several restored scripts are historical originals. Some still retain `/mnt/data/...` fallbacks. The main field-core scripts were patched to prefer repo-local paths.
