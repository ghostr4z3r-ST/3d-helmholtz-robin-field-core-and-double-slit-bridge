# Reproducibility status

## Scope
Repository 2 assumes the validated field framework of Paper 1 / Repository 1. It does not repeat the foundational artifact-hardening argument from scratch.

## Current public-core state
- **179** tracked public-core artifacts
- **177** recovered public-core artifacts currently present
- **2** tracked public-core artifacts still missing

## Present recovered highlights
- Dense beta tracking and field-core robustness outputs are present.
- Opposite / carrier / CEF diagnostics are effectively complete.
- Reduction-to-minimal-core and field-core-function blocks are present in strong form.
- Double-slit, detector matrix, and effective detector law blocks are present in strong form.
- Microscopic Robin-noise support is now present with summary, curves, coefficients, and report note.

## Remaining public-core gaps

## Not counted as public-core gap
- `build_atom_H.py` is treated as historical builder/provenance material rather than as a required public reproducibility artifact.

## Technical note
Several restored scripts are historical originals. Some still retain `/mnt/data/...` path defaults. The field-core scripts were patched to prefer repo-local paths with `/mnt/data` as fallback.


## Reconstructions

- `scripts/06_double_slit_baselines/one_slit_baseline_robin_like.py` is included as a transparent best-effort reconstruction from the recovered baseline CSV/report outputs.
- `figures/04_reduction_to_minimal_core/stepE_kernel_plus_bridge_accuracy.png` was regenerated from the recovered `stepD_stepE_bridge_geometry.py` logic and the surviving `stepE_kernel_plus_bridge_summary.csv`.
