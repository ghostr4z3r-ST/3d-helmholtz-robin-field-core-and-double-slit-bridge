# Smoke test notes

A minimum smoke reproduction is available via:

```bash
bash scripts/reproduce_minimum.sh
```

The current smoke path exercises:
- dense beta branch tracking
- subspace rotation robustness
- axis permutation robustness
- readout robustness

Known note: `readout_robustness.py` requires `--pts-per-cell >= 3`.
