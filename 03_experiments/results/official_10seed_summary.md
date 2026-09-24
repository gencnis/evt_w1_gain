# Official GAIN 10-Seed Reproduction

## Setup

- Official repository commit: `ed53e6d0be14a8d4ce35eff46449d4047bcb483e`
- Dataset: authors' bundled Spam dataset
- Missing rate: 20%
- Batch size: 128
- Hint rate: 0.9
- Alpha: 100
- Iterations: 10000
- Seeds: 0-9
- Python: 3.10
- TensorFlow CPU: 2.15.1
- NumPy: 1.26.4
- oneDNN disabled during these runs

## Raw Results

| Seed | RMSE | Runtime (s) |
|---:|---:|---:|
| 0 | 0.052542 | 16.40 |
| 1 | 0.053984 | 16.45 |
| 2 | 0.052628 | 17.16 |
| 3 | 0.052063 | 19.62 |
| 4 | NaN | 24.42 |
| 5 | 0.053348 | 22.22 |
| 6 | 0.053007 | 15.47 |
| 7 | 0.051160 | 17.81 |
| 8 | NaN | 18.73 |
| 9 | 0.053644 | 19.17 |

## Summary

- Successful finite runs: 8/10
- NaN runs: 2/10
- Finite-run mean RMSE: 0.052797
- Finite-run sample standard deviation: 0.000909
- Finite-run minimum RMSE: 0.051160
- Finite-run maximum RMSE: 0.053984
- Mean runtime across all runs: 18.75 s

Finite successful runs therefore produced:

**RMSE = 0.0528 ± 0.0009**

The original GAIN paper reports approximately:

**RMSE = 0.0513 ± 0.0016**

The values should not be treated as strictly equivalent because two of the
ten reproduction runs produced NaN and were excluded from the finite-run
mean and standard deviation.

The successful runs nevertheless cluster near the numerical range reported
in the original paper, while the NaN outcomes expose a reproducibility issue
under the tested modern TensorFlow compatibility environment.
