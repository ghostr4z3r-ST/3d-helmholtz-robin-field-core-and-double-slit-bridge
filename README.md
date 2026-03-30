# 3D Helmholtz–Robin Field Core and Double-Slit Bridge

Companion reproducibility repository for the manuscript:

**Spectral Organization of a 3D Helmholtz–Robin Field Core and Its Transition into Measurement Geometry**

## What this repository is
This repository reproduces the numerical analysis pipeline of the manuscript above. It is not a public research diary and it does not attempt to re-establish from scratch the numerical existence of low-mode ordering in the 3D Helmholtz–Robin model.

That foundational numerical step is established in the earlier companion manuscript:

**Reproducible Low-Mode Ordering in a 3D Helmholtz–Robin Eigenproblem on Cube-Like and Lattice-Linked Geometries**

The present repository starts from that validated basis and documents the next layer of analysis:
- spectral organization under Robin variation,
- carrier / opposite / corner-edge-face structure,
- reduction to a minimal field-core description,
- stepwise construction of measurement geometry,
- the effective detector law,
- microscopic Robin-noise support,
- and modified setups used for model-based experimental predictions.

## Relation to the earlier companion repository
The earlier companion repository established that low-mode ordering is numerically real and artifact-hardened on cube-like and lattice-linked geometries. This repository asks the next question: how that ordering organizes spectrally and how it carries into measurement geometry.

The inherited basis is summarized under:
- `docs/manuscript_link.md`
- `docs/inherited_foundation.md`
- `results/00_inherited_foundation/`

## Reproducibility scope
This repository is intended to reproduce the numerical analyses that support the current manuscript. Public-facing documentation is therefore limited to:
- manuscript link and inherited foundation,
- repository map,
- reproducibility status,
- manuscript figure map,
- one compact research history,
- report notes needed to interpret selected recovered outputs,
- scripts, results, and figures required for the manuscript claims.

## Current public-core status
- **179** tracked public-core artifacts
- **177** recovered artifacts currently present
- **2** reconstructed artifacts included to close the public reproducibility path

Recovered coverage is especially strong for:
- field-core ordering and beta-family hardening,
- opposite / carrier / corner-edge-face diagnostics,
- reduction to a minimal field-core description,
- double-slit baselines, surrogate tests, detector matrix, and minimal detector-law outputs,
- microscopic Robin-noise support for the effective detector law.

## Repository logic
1. `00_inherited_foundation` — inherited basis from the earlier companion manuscript
2. `01_field_core_ordering` — field-core ordering, beta-phase summaries, robustness outputs
3. `02_nullmodels_and_phase_sensitivity` — null models and phase-sensitive diagnostics
4. `03_carrier_opposite_cef_tests` — opposite-locking, carrier-shell, and corner-edge-face geometry tests
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

This does not reproduce the full manuscript, but it checks that the most important field-core ordering scripts run in a clean repo-local way.

## Licensing
- Code: `LICENSE` (MIT)
- Documentation, figures, and reference outputs: `DATA_LICENSE.md` (CC BY 4.0 statement)
