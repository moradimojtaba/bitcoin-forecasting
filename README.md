# Horizon-Dependent Bitcoin Return Forecasting

**Evidence from Adaptive Forecast Combination and Rigorous Out-of-Sample Evaluation**

Mojtaba Moradi

## Overview

This repository contains the computational materials associated with the study of Bitcoin return forecasting across multiple forecast horizons. The central question is whether forecasting performance depends on the horizon and whether a transparent, recency-adaptive combination of heterogeneous forecasts can improve out-of-sample return prediction.

The experiment evaluates **13 forecasting specifications** across **eight direct horizons**, from 15 days to 36 months, using an expanding-window walk-forward design. The target is the direct cumulative log return:

`R(t,h) = 100 * log(P[t+h] / P[t])`

## ⚠️ IMPORTANT: the full program is computationally heavy

The publication-grade experiment is a **large, time-consuming and computationally intensive run**. The neural-network models are repeatedly retrained over thousands of forecast origins. This is not a quick demonstration program.

On the recorded computing environment, the complete primary experiment required approximately **12 hours** in total, although runtime varies substantially by horizon and hardware. CPU/GPU activity can fluctuate during model fitting, memory management, and disk operations; temporary low CPU usage should not automatically be interpreted as a frozen process.

### If the program is interrupted: RUN IT AGAIN

**Do not delete the checkpoint files. Do not start by deleting the results directory. Simply run the program again.**

The forecasting program contains a checkpoint/resume mechanism specifically for this long computation. Partial results are saved during execution. On restart, the program checks the existing checkpoint against the dataset fingerprint and the forecasting configuration. Compatible completed forecast origins are reused, the causal out-of-sample error history needed by the Adaptive Ensemble is reconstructed, and the program continues with the remaining origins.

Therefore:

1. Start the program normally.
2. Allow it to run for as long as necessary.
3. If Windows, Python, TensorFlow, power loss, sleep/hibernate, or another problem interrupts it, **leave the generated checkpoint files untouched**.
4. Run the same command again.
5. The program will **not blindly start the interrupted horizon from the beginning**; it will reuse compatible completed calculations and continue from the remaining forecast origins.

This behavior is an intentional part of the reproducibility design.

> **For exact reproduction, do not change the dataset or core forecasting configuration between runs.** A configuration or dataset change can intentionally invalidate an old checkpoint.

## Repository structure

```text
bitcoin-forecasting/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── run_forecasting.bat
│
├── code/
│   ├── README.md
│   └── bitcoin_forecasting.py
│
├── data/
│   ├── README.md
│   └── processed/
│       └── bitcoin_modeling_dataset.csv
│
├── results/
│   ├── README.md
│   ├── forecasting_results_V6_2_multihorizon.csv.gz
│   ├── forecasting_metrics_V6_2_multihorizon.csv
│   ├── computational_audit_V6_2.csv
│   ├── metadata_V6_2.json
│   ├── model_confidence_set_sensitivity_V6_2.csv
│   ├── diebold_mariano_pairwise_HAC_V6_2.csv
│   ├── dm_adaptive_vs_best_nonensemble_V6_2.csv
│   ├── sign_distribution_tests_V6_2.csv
│   ├── stationarity_ADF_KPSS_V6_2.csv
│   ├── stability_CUSUM_V6_2.csv
│   └── ablation_summary_V6_2.csv
│
├── figures/
│   ├── figure1_rmse_horizon.png
│   ├── figure2_ae_improvement.png
│   ├── figure3_directional_base_rate.png
│   ├── figure4_ensemble_weights.png
│   └── figure5_recent_robustness.png
│
└── paper/
    ├── README.md
    ├── bitcoin_paper_Q1_refined_FINAL.tex
    └── bitcoin_paper_Q1_refined_FINAL.pdf
```

## Experimental design

- Initial training window: **1,000 observations**
- Retraining interval: **25 forecast origins**
- Random seed: **20260819**
- Forecast origins: **all valid origins**
- LSTM lookback: **30 observations**
- Maximum LSTM epochs: **40**
- Early stopping patience: **6**
- Batch size: **32**
- ARIMA order: **(1,0,1)**
- GLSAR: `rho=1`, maximum 8 iterations
- Adaptive Ensemble history: **20 previous OOS errors**
- MCS bootstrap repetitions: **1,000**
- Robustness bootstrap repetitions: **10,000**

The frozen modeling dataset contains **4,761 observations from 2011-05-14 through 2024-05-25**. Its SHA-256 fingerprint is recorded in `results/metadata_V6_2.json`.

## Forecast horizons

| Label | Days |
|---|---:|
| 15D | 15 |
| 1M | 30 |
| 2M | 60 |
| 3M | 90 |
| 6M | 180 |
| 12M | 365 |
| 24M | 730 |
| 36M | 1095 |

The number of valid forecast origins decreases at longer horizons because the sample is finite and the targets are constructed directly for each horizon.

## Models

1. Zero Return
2. Historical Mean
3. Random Walk with Drift
4. Historical Median
5. Majority Direction
6. Persistence Direction
7. ARIMA
8. OLS
9. GLS
10. LSTM
11. VS-LSTM
12. GLS-LSTM
13. Adaptive Ensemble

Random Walk with Drift is excluded from the magnitude ensemble because it is identical to Historical Mean under the direct-return formulation. Majority Direction and Persistence Direction are direction-only benchmarks and are therefore not magnitude components of the ensemble.

## Adaptive Ensemble

The Adaptive Ensemble combines nine magnitude forecasts using a transparent recency-adaptive rule:

- Zero Return
- Historical Mean
- Historical Median
- ARIMA
- OLS
- GLS
- LSTM
- VS-LSTM
- GLS-LSTM

After an initial history of 20 eligible forecasts, weights are proportional to the inverse of each model's recent mean absolute error. Only **earlier out-of-sample errors** are used to construct the current weights, preserving causality.

The ensemble should not be interpreted as a universally optimal or permanently superior model.

## Statistical evaluation

The repository contains outputs for:

- return RMSE and MAE;
- directional accuracy, balanced accuracy, F1, and MCC;
- overlap-aware HAC Diebold–Mariano comparisons;
- a block-bootstrap Hansen-style model-set diagnostic;
- positive-return/sign-distribution analysis;
- CUSUM forecast-error stability diagnostics;
- ADF/KPSS target diagnostics;
- recent-sample and non-overlap robustness diagnostics;
- post-hoc equal-weight ablation using stored OOS forecasts;
- computational and reproducibility audits.

The model-set procedure is deliberately described as a **Hansen-style bootstrap approximation/diagnostic**, not as an exact implementation of the canonical MCS algorithm.

## Installation

### 1. Clone the repository

```bash
git clone <REPOSITORY-URL>
cd bitcoin-forecasting
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The recorded primary environment used Python 3.13.2, NumPy 2.3.3, pandas 2.3.3, SciPy 1.16.2, statsmodels 0.14.6, and TensorFlow 2.21.0. The code can run on CPU; GPU availability depends on the local TensorFlow/platform configuration.

## Running the experiment

Place the exact frozen dataset at:

```text
data/processed/bitcoin_modeling_dataset.csv
```

Then run:

```bash
python code/bitcoin_forecasting.py
```

or, on Windows:

```text
run_forecasting.bat
```

### Very important during a full run

- Connect the computer to AC power.
- Prevent Windows from entering sleep/hibernate.
- Do not delete checkpoint files.
- Do not repeatedly stop and restart the program unnecessarily.
- If an interruption occurs, **run the same command again** and allow the resume mechanism to continue the work.

## Checkpoint and resume behavior

Checkpoints are stored under the runtime results directory. They contain completed forecast-origin/model records and a configuration fingerprint.

The resume logic validates:

- dataset SHA-256 fingerprint;
- horizon;
- target alignment;
- training/retraining configuration;
- model specification;
- feature specification;
- ensemble specification;
- checkpoint schema.

For partial checkpoints, the program keeps only the contiguous completed prefix and reconstructs the prior OOS errors used by the Adaptive Ensemble. This prevents a restart from accidentally using information from future forecast origins.

## Interpretation of the results

The associated manuscript emphasizes several qualifications:

- The empirical model ranking is **horizon-dependent**.
- The Adaptive Ensemble is a forecasting strategy, not a claim of universal superiority.
- Very high long-horizon directional accuracy must be interpreted against the unconditional positive-return base rate.
- At 36 months, approximately 99.68% of realized returns are positive, while the Adaptive Ensemble reaches approximately 99.81% directional accuracy; its excess over an Always-Up rule is only about 0.13 percentage points.
- Lower RMSE does not imply trading profitability.
- Overlapping long-horizon observations reduce the amount of effectively independent information; HAC and block-bootstrap diagnostics are therefore important.
- Non-overlapping samples become extremely small at long horizons and are used as diagnostics rather than as standalone confirmation.

## Data provenance

The modeling data are a frozen historical dataset derived from Bitcoin price and blockchain-related observations used in the study. The exact input fingerprint and sample range are recorded in the metadata.

Before the repository is treated as the definitive public replication archive, the exact source providers, source URLs, retrieval dates, cleaning/synchronization procedures, transformations, and third-party data licensing terms should be documented here or in `docs/data_provenance.md`.

## Citation

Please cite the associated manuscript and this repository when using the code, data, or results.

## Author

**Mojtaba Moradi** — Department of Industrial Engineering, University of Guilan


