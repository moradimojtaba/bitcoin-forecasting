# Results

This directory contains the numerical outputs of the full forecasting experiment:

- forecast-level out-of-sample predictions;
- multihorizon model metrics;
- pairwise HAC Diebold–Mariano comparisons;
- Adaptive Ensemble versus best non-ensemble comparisons;
- model-set sensitivity diagnostics;
- sign-distribution diagnostics;
- stationarity and CUSUM diagnostics;
- post-hoc ablation results;
- computational audit and reproducibility metadata.

The stored forecast-level panel can be used to reconstruct the principal RMSE, MAE, and directional metrics without retraining the neural networks.

The forecast-level panel is stored as `forecasting_results_V6_2_multihorizon.csv.gz` to keep the public repository manageable. It is a standard gzip-compressed CSV and can be decompressed with 7-Zip, WinRAR, Python, or other standard tools.
