#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/_reproduce_minimum"
mkdir -p "$OUT/01_field_core_ordering"
python "$ROOT/scripts/01_field_core_ordering/dense_beta_branch_tracking.py" \
  --betas 0,1,2 --pts-per-cell 3 --modes 8 \
  --full-csv "$OUT/01_field_core_ordering/dense_beta_branch_tracking_full.csv" \
  --winners-csv "$OUT/01_field_core_ordering/dense_beta_branch_tracking_winners.csv" \
  --summary-csv "$OUT/01_field_core_ordering/dense_beta_branch_tracking_summary.csv" \
  --branch-summary-csv "$OUT/01_field_core_ordering/dense_beta_branch_tracking_branch_summary.csv" \
  --plot "$OUT/01_field_core_ordering/dense_beta_branch_tracking_consensus.png"
python "$ROOT/scripts/01_field_core_ordering/subspace_rotation_robustness.py" \
  --betas 0,1.5 --pts-per-cell 3 --modes 8 --trials 4 \
  --full-csv "$OUT/01_field_core_ordering/subspace_rotation_robustness_full.csv" \
  --summary-csv "$OUT/01_field_core_ordering/subspace_rotation_robustness_summary.csv" \
  --heatmap-png "$OUT/01_field_core_ordering/subspace_rotation_robustness_consensus_family.png" \
  --q-heatmap-png "$OUT/01_field_core_ordering/subspace_rotation_robustness_consensus_q.png"
python "$ROOT/scripts/01_field_core_ordering/axis_permutation_robustness.py" \
  --betas 0,1.5 --pts-per-cell 3 --modes 8 \
  --full-csv "$OUT/01_field_core_ordering/axis_permutation_robustness_full.csv" \
  --summary-csv "$OUT/01_field_core_ordering/axis_permutation_robustness_summary.csv" \
  --family-heatmap-png "$OUT/01_field_core_ordering/axis_permutation_robustness_consensus_family.png" \
  --q-heatmap-png "$OUT/01_field_core_ordering/axis_permutation_robustness_consensus_q.png"
python "$ROOT/scripts/01_field_core_ordering/readout_robustness.py" \
  --betas 0,1.5 --pts-per-cell 3 --modes 8 \
  --full-csv "$OUT/01_field_core_ordering/readout_robustness_full.csv" \
  --winners-csv "$OUT/01_field_core_ordering/readout_robustness_winners.csv" \
  --summary-csv "$OUT/01_field_core_ordering/readout_robustness_summary.csv" \
  --family-plot "$OUT/01_field_core_ordering/readout_robustness_family.png" \
  --q-plot "$OUT/01_field_core_ordering/readout_robustness_q.png"
echo "Minimum reproduction finished under $OUT"
