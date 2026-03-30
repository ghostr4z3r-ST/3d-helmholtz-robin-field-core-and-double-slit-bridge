# 3D Helmholtz–Robin Field Core and Double-Slit Bridge (Paper 2 Companion Repository)

Companion reproducibility repository for **Paper 2** of the Helmholtz–Robin project.

## What this repository is
This repository is the **second-stage computational repository** of the project. It does **not** start from a blank model and it does **not** attempt to re-establish the numerical existence of ordering from scratch. That foundational step belongs to **Paper 1 / Repository 1**.

Repository 2 begins only after the Paper-1 result is taken as established:
- a 3D Helmholtz–Robin field model on cube-like and lattice-linked geometries,
- artifact-hardened ordering in that model,
- the basic field/readout vocabulary,
- and the null-model discipline used to separate robust structure from easy artifacts.

From that validated basis, Repository 2 pursues the next question:
**how the Paper-1 ordering organizes spectrally under Robin variation and how that field-core organization carries into the stepwise double-slit and detector build-up.**

## Relation to Paper 1 / Repository 1
**Paper 1 / Repository 1 established that the ordering is numerically real.**
**Paper 2 / Repository 2 studies how that ordering is spectrally organized and how it extends into measurement geometry.**

This means:
- Repository 1 is the numerical-methodological foundation.
- Repository 2 is the field-core, spectral-organization, and double-slit continuation.
- Repository 2 should therefore be read as a **bridge repository**, not as an independent starting point.

The inherited basis from Paper 1 is documented under:
- `docs/paper1_dependency_map.md`
- `results/00_foundation_from_paper1/`
- `docs/forschungshistorie_paper2.md`

## Current recovery status
Tracked **public-core** Paper-2 artifacts at the current release-candidate build:
- **179** tracked artifacts
- **177** recovered artifacts currently present
- **2** tracked artifacts still missing from the public core

Recovered blocks with especially strong coverage:
- field-core ordering and beta-family hardening outputs
- opposite / carrier / CEF diagnostics
- reduction to a minimal field-core description
- double-slit baselines, surrogate tests, detector matrix, and minimal detector-law outputs
- microscopic Robin-noise support for the effective detector law

## What is still missing from the public core

A historical builder-side script (`build_atom_H.py`) is **not treated as part of the public reproducibility core**. It belongs, if recovered at all, to the earlier visualization/provenance phase rather than to the reproducible claim path of Paper 2.

## Repository logic
1. `00_foundation_from_paper1` — explicit dependency bridge from Paper 1 / Repo 1
2. `01_field_core_ordering` — field-core ordering, beta-phase summaries, robustness outputs
3. `02_nullmodels_and_phase_sensitivity` — null models and phase-sensitive diagnostics
4. `03_carrier_opposite_cef_tests` — opposite-locking, carrier-shell, and CEF geometry tests
5. `04_reduction_to_minimal_core` — minimal kernel, carrier graph, bridge reduction
6. `05_field_core_function` — transition-zone and field-core-function diagnostics
7. `06_double_slit_baselines` — one-slit and coherent double-slit baselines
8. `07_measurement_surrogates_and_robin_tests` — disturbance and Robin-variation tests
9. `08_real_detector_model` — composite detector model
10. `09_detector_matrix_and_minimal_formula` — detector matrix and effective detector law
11. `10_microscopic_robin_noise_model` — microscopic continuation of the detector law

## Running a minimum smoke reproduction
A small public smoke reproduction is provided via:

```bash
bash scripts/reproduce_minimum.sh
```

This does **not** reproduce the full paper, but it checks that the most important field-core ordering scripts run in a clean repo-local way.

## Status note
This is a **public-release candidate** for the future public Paper-2 repository. It already contains recovered scripts, reference CSVs, figures, report notes, and provenance material, but a very small number of public-core artifacts remain missing and many historical scripts still reflect their original exploratory origins.
