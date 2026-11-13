# suicide-prediction-japan-jmir
Processed datasets and analytical source code for "Interpreting the Role of Digital and Socioeconomic Signals in Suicide Prediction: A Time-Series and Machine Learning Study of Monthly Data in Japan"
# Interpreting the Role of Digital and Socioeconomic Signals in Suicide Prediction: A Time-Series and Machine Learning Study of Monthly Data in Japan

This repository contains the processed datasets and analytical source code for the study:
> **"Interpreting the Role of Digital and Socioeconomic Signals in Suicide Prediction: A Time-Series and Machine Learning Study of Monthly Data in Japan"**

## Repository Structure

### 1. `data/` (Processed Datasets)
Monthly time-series datasets covering January 2009 to December 2024 ($N=192$ months), including official monthly suicide counts, Google Trends search interest indices, macroeconomic indicators, and meteorological variables:
- `data_japan_all_get260813.csv`: Nationwide Japan dataset.
- `data_tokyo_all_get260819.csv`: Tokyo dataset without meteorological variables.
- `data_tokyo_plusMET_get260819.csv`: Tokyo dataset with meteorological variables.

### 2. `code/` (Analytical and Visualization Scripts)
- `code_sarimax_1by1_gs.py`: Walk-forward validation pipeline for SARIMA and SARIMAX models (stationarity tests, 1-by-1 Granger causality screening, VIF filtering, AIC grid search, monthly refitting, and parameter/diagnostic exports).
- `code_analyze.py`: Evaluation pipeline for SARIMA/SARIMAX models (MAE, RMSE, MASE, two-sided HLN-DM tests, Supplement Tables S2–S4, and Figure 2 trajectory plots).
- `requirements_sarimax` : Requirements for SARIMAX pipelines
- `code_lightGBM_TPE_NI_10seed.py`: Walk-forward validation pipeline for LightGBM Models A–D across 10 independent random seeds (correlation filter, sample-weighted null importance screening, and Optuna TPE regularization tuning).
- `code_analyze_10seed.py`: Ensemble evaluation pipeline for LightGBM models (10-seed performance metrics, pairwise two-sided HLN-DM tests, stability metrics [Pearson correlation and Jaccard index], Supplement Tables S6–S7, and Figure 3 trajectory plots).
- `code_shap_10seed.py`: Global and local SHAP analysis pipeline with a 90% cumulative importance cutoff (Tables 4 and 5).
- `requirements_lightGBM` : Requirements for lightGBM pipelines


## Requirements
- Python 3.11 for SARIMAX
- Python 3.13 for lightGBM
