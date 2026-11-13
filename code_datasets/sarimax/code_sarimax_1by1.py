# ==============================================================================
# code_sarimax_1by1_gs_01.py: Dynamic Feature Selection with 1-by-1 Lag Approach
# ==============================================================================
"""
This script performs Walk-Forward Validation for SARIMA/SARIMAX models.
It dynamically selects the optimal lag for each external feature using
Granger Causality tests, handles multicollinearity via VIF, and exports
model specifications (AIC/BIC), internal AR/MA parameter summaries,
and exogenous feature importance summaries.
"""

# --- 1. Library Imports & Configurations ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tools.sm_exceptions import InfeasibleTestError
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os
import datetime
import sys
import shutil
from numpy.linalg import LinAlgError
import importlib.metadata
import re

# Utility class to duplicate print outputs to console and log file
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# --- 2. Environment Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]
os.chdir(script_dir)

if len(sys.argv) < 2:
    print("★★★ USAGE ERROR: Please provide the data file name as a command-line argument. ★★★")
    print(f"Example: python {script_name}.py data_japan_all.csv")
    sys.exit(1)

file_name = sys.argv[1]
match = re.search(r'data_([^.]+)', file_name)
region_info = match.group(1) if match else "unknown"

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
output_folder = f'{script_name}_{region_info}_output_{timestamp}'
os.makedirs(output_folder, exist_ok=True)

try:
    current_script_path = os.path.abspath(__file__)
    shutil.copy2(current_script_path, output_folder)
except Exception:
    pass

# --- 3. Logging & Warning Management ---
log_file_path = os.path.join(output_folder, 'execution_log.txt')
log_file = open(log_file_path, 'w', encoding='utf-8')
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, log_file)
sys.stderr = sys.stdout

# Route warnings to a separate file to keep stdout clean
warnings_log_path = os.path.join(output_folder, 'warnings_log.txt')
warnings_file = open(warnings_log_path, 'w', encoding='utf-8')

def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    log_message = f"[{category.__name__}] {message} (File: {filename}, Line: {lineno})\n"
    warnings_file.write(log_message)
    warnings_file.flush()

warnings.showwarning = custom_showwarning
warnings.filterwarnings('always')
warnings.filterwarnings('ignore', category=FutureWarning)

print(f"--- Analysis started at {timestamp} ---")
print(f"--- Target data: {file_name} ---")

# --- 4. Helper Functions ---
def save_environment_info(output_dir):
    """
    Saves the current Python environment (installed packages and versions)
    to a requirements_env.txt file for reproducibility.
    """
    env_path = os.path.join(output_dir, 'requirements_env.txt')
    try:
        with open(env_path, 'w', encoding='utf-8') as f:
            dists = sorted(importlib.metadata.distributions(), key=lambda x: str(x.metadata.get('Name', '')).lower())
            for dist in dists:
                f.write(f"{dist.metadata['Name']}=={dist.version}\n")
    except Exception as e:
        print(f"Warning: Failed to save environment info. {e}")

save_environment_info(output_folder)

def param_sort_key(param_name):
    """Sorts internal parameters in standard order: ar.L*, ma.L*, ar.S.L*, ma.S.L*."""
    s = str(param_name)
    m_ar = re.match(r'^ar\.L(\d+)$', s)
    if m_ar: return (1, int(m_ar.group(1)), s)
    m_ma = re.match(r'^ma\.L(\d+)$', s)
    if m_ma: return (2, int(m_ma.group(1)), s)
    m_sar = re.match(r'^ar\.S\.L(\d+)$', s)
    if m_sar: return (3, int(m_sar.group(1)), s)
    m_sma = re.match(r'^ma\.S\.L(\d+)$', s)
    if m_sma: return (4, int(m_sma.group(1)), s)
    if s in ('intercept', 'drift'): return (5, 0, s)
    return (6, 0, s)


# --- 5. Main Execution Block ---
try:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = [15, 8]

    VIF_THRESHOLD = 5.0
    GRANGER_P_VALUE_THRESHOLD = 0.05
    VALIDATION_START_YEAR = 2022
    TRAINING_START_DATE = '2009-01-01'

    # Discrete lag space for exogenous variables.
    # Real-time variables (lagcap1_*) can utilize lag=1 at forecast origin.
    EXOG_LAGS = [1, 2, 3, 4, 6, 12, 24]

    df_raw = pd.read_csv(file_name)
    date_col_name = df_raw.columns[0]
    df_raw[date_col_name] = pd.to_datetime(df_raw[date_col_name])
    df = df_raw.set_index(date_col_name).asfreq('MS')

    target_col = df.columns[0]
    exog_cols = df.columns[1:].tolist()
    y_full = df[target_col]

    # --- 6. Walk-Forward Validation ---
    validation_index = df[df.index.year >= VALIDATION_START_YEAR].index
    predictions_sarima, predictions_sarimax, actuals = [], [], []

    sarima_order, sarima_seasonal_order = None, None
    sarimax_order, sarimax_seasonal_order = None, None
    selected_feature_cols = []
    current_vars_meta = {}
    X_lagged_full = pd.DataFrame(index=df.index)

    # Storage lists for logs and supplementary tables
    info_criteria_results = []
    annual_selected_features_list = []
    monthly_exog_stats_list = []
    monthly_internal_params_list = []
    monthly_diagnostics_list = []

    def extract_diagnostics(arima_model, model_label, dt):
        res = arima_model.arima_res_
        # statsmodels標準の診断メソッド（d=1の初期バーンインを除いた標準化残差に対して計算）
        # Ljung-Box: lags=12を指定し、ラグ1(index 0)とラグ12(index 11)の両方を取得
        lb_res = res.test_serial_correlation(method='ljungbox', lags=12)[0]
        lb_q_lag1, lb_p_lag1 = float(lb_res[0, 0]), float(lb_res[1, 0])
        lb_q_lag12, lb_p_lag12 = float(lb_res[0, 11]), float(lb_res[1, 11])
        jb_res = res.test_normality(method='jarquebera')[0]
        jb_stat, jb_p = float(jb_res[0]), float(jb_res[1])
        return {
            'Date': dt.strftime('%Y-%m-%d'), 'Year': dt.year, 'Model': model_label,
            'AIC': round(float(res.aic), 2), 'BIC': round(float(res.bic), 2),
            'LB_Q_Lag1': round(lb_q_lag1, 2), 'LB_P_Lag1': round(lb_p_lag1, 4), 'LB_Pass_Lag1': lb_p_lag1 > 0.05,
            'LB_Q_Lag12': round(lb_q_lag12, 2), 'LB_P_Lag12': round(lb_p_lag12, 4), 'LB_Pass_Lag12': lb_p_lag12 > 0.05,
            'JB_Stat': round(jb_stat, 2), 'JB_P': round(jb_p, 4), 'JB_Pass': jb_p > 0.05
        }

    def extract_internal_params(arima_model, model_label, dt, exog_cols_used=None):
        """Extracts internal AR, MA, seasonal AR/MA, intercept, and variance parameters."""
        if exog_cols_used is None:
            exog_cols_used = []
        res = arima_model.arima_res_
        records = []
        for param_name in res.params.index:
            if param_name in exog_cols_used:
                continue
            if param_name.startswith(('ar.', 'ma.')):
                param_type = 'AR_MA'
            elif param_name in ('intercept', 'drift'):
                param_type = 'Intercept'
            elif param_name == 'sigma2':
                param_type = 'Variance'
            else:
                param_type = 'Other'

            coef = float(res.params[param_name])
            se = float(res.bse[param_name]) if param_name in res.bse.index else np.nan
            z_val = float(res.zvalues[param_name]) if param_name in res.zvalues.index else np.nan
            p_val = float(res.pvalues[param_name]) if param_name in res.pvalues.index else np.nan

            records.append({
                'Date': dt.strftime('%Y-%m-%d'),
                'Year': dt.year,
                'Model': model_label,
                'Parameter': param_name,
                'Param_Type': param_type,
                'Coef': coef,
                'Std_Error': se,
                'Z_Stat': z_val,
                'Abs_Z_Stat': abs(z_val) if pd.notna(z_val) else np.nan,
                'P_Value': p_val,
                'Significant_005': (p_val < 0.05) if pd.notna(p_val) else False
            })
        return records


    for current_date in validation_index:
        train_end_date = current_date - pd.DateOffset(months=2)

        # ======================================================================
        # 6A. Yearly Update: Feature Selection & Order Search (Every January)
        # ======================================================================
        if current_date.month == 1 or sarima_order is None:
            print(f"\n===== Yearly Update for {current_date.year} (using data up to {train_end_date.strftime('%Y-%m')}) =====")

            fs_train_data = df[(df.index >= TRAINING_START_DATE) & (df.index <= train_end_date)].copy()
            y_fs = fs_train_data[target_col]
            y_fs_stationary = y_fs.diff(1).dropna()

            # 1. Pre-build shifted feature matrix and perform 1-by-1 Granger causality tests
            X_lagged_full = pd.DataFrame(index=df.index)
            promising_results = []

            for col in exog_cols:
                raw_series_train = fs_train_data[col].dropna()
                if raw_series_train.nunique() <= 1:
                    continue

                # Stationarity check (ADF) strictly within training window
                p_val_adf = adfuller(raw_series_train)[1]
                is_differenced = p_val_adf > 0.05

                if is_differenced:
                    x_train_stat = raw_series_train.diff(1).dropna()
                    if len(x_train_stat) <= 20 or adfuller(x_train_stat)[1] > 0.05:
                        continue
                    full_series_processed = df[col].diff(1)
                else:
                    full_series_processed = df[col]

                parts = col.split('_')
                try:
                    min_lag = int(parts[0][6:])
                except (ValueError, IndexError):
                    raise ValueError(f"Could not parse lag cap from column name: '{col}'.")

                lag_range = [lag for lag in EXOG_LAGS if lag >= min_lag]

                best_p, best_lag, best_col_name = 1.0, 0, None
                for lag in lag_range:
                    col_lag_name = f"{col}_lag{lag:02}"
                    # Align data Available for predicting month t at row t
                    X_lagged_full[col_lag_name] = full_series_processed.shift(lag)

                    # Granger causality test (maxlag=1 shifts the 2nd column by 1 internally, so pass shift(lag-1))
                    try:
                        x_for_gc = full_series_processed.loc[fs_train_data.index].shift(lag - 1)
                        test_df = pd.concat([y_fs_stationary, x_for_gc], axis=1).dropna()
                        if len(test_df) < 20 or test_df.iloc[:, 1].nunique() <= 1:
                            continue

                        gc_results = grangercausalitytests(test_df, maxlag=1)
                        p_val = gc_results[1][0]['ssr_ftest'][1]

                        if p_val < best_p:
                            best_p, best_lag, best_col_name = p_val, lag, col_lag_name
                    except (InfeasibleTestError, ValueError, LinAlgError, KeyError):
                        continue

                if best_p < GRANGER_P_VALUE_THRESHOLD and best_col_name is not None:
                    promising_results.append({
                        'Variable': col,
                        'Feature_Col': best_col_name,
                        'Best_Lag': best_lag,
                        'Granger_P_Value': best_p,
                        'Is_Differenced': is_differenced
                    })

            # 2. Multicollinearity filtering via VIF
            if promising_results:
                final_vars_df = pd.DataFrame(promising_results).sort_values(by='Granger_P_Value').copy()
            else:
                final_vars_df = pd.DataFrame(columns=['Variable', 'Feature_Col', 'Best_Lag', 'Granger_P_Value', 'Is_Differenced'])

            final_vif_map = {}
            if len(final_vars_df) == 1:
                final_vif_map[final_vars_df.iloc[0]['Feature_Col']] = 1.0
            elif len(final_vars_df) > 1:
                while len(final_vars_df) >= 2:
                    current_cols = final_vars_df['Feature_Col'].tolist()
                    X_vif_df = X_lagged_full.loc[fs_train_data.index, current_cols].dropna()
                    if len(X_vif_df) < 20:
                        break

                    X_vif_const = sm.add_constant(X_vif_df)
                    vif_series = pd.Series(
                        [variance_inflation_factor(X_vif_const.values, i + 1) for i in range(len(current_cols))],
                        index=current_cols
                    )

                    if vif_series.max() > VIF_THRESHOLD:
                        worst_col = vif_series.idxmax()
                        original_var_name = worst_col.rsplit('_lag', 1)[0]
                        final_vars_df = final_vars_df[final_vars_df['Variable'] != original_var_name]
                        if len(final_vars_df) == 1:
                            final_vif_map[final_vars_df.iloc[0]['Feature_Col']] = 1.0
                    else:
                        final_vif_map = vif_series.to_dict()
                        break

            selected_feature_cols = final_vars_df['Feature_Col'].tolist()
            current_vars_meta = {}

            if not final_vars_df.empty:
                final_vars_df['Final_VIF'] = final_vars_df['Feature_Col'].map(final_vif_map)
                print(f"Selected Exogenous Features for {current_date.year} ({len(selected_feature_cols)} features):")
                print(final_vars_df[['Variable', 'Best_Lag', 'Is_Differenced', 'Granger_P_Value', 'Final_VIF']].to_string(index=False))

                for _, row in final_vars_df.iterrows():
                    f_col = row['Feature_Col']
                    current_vars_meta[f_col] = {
                        'Variable': row['Variable'],
                        'Best_Lag': int(row['Best_Lag']),
                        'Is_Differenced': bool(row['Is_Differenced']),
                        'Granger_P_Value': float(row['Granger_P_Value']),
                        'Final_VIF': float(row['Final_VIF']) if pd.notna(row['Final_VIF']) else np.nan
                    }
                    annual_selected_features_list.append({
                        'Year': current_date.year,
                        'Variable': row['Variable'],
                        'Feature_Col': f_col,
                        'Best_Lag': int(row['Best_Lag']),
                        'Is_Differenced': bool(row['Is_Differenced']),
                        'Granger_P_Value': round(float(row['Granger_P_Value']), 4),
                        'Final_VIF': round(float(row['Final_VIF']), 2) if pd.notna(row['Final_VIF']) else np.nan
                    })
            else:
                print(f"[WARNING] No exogenous variables passed Granger & VIF screening for {current_date.year}.")
                print("          SARIMAX will reduce to the baseline SARIMA specification for this year.")
                annual_selected_features_list.append({
                    'Year': current_date.year,
                    'Variable': 'None',
                    'Feature_Col': 'None',
                    'Best_Lag': np.nan,
                    'Is_Differenced': np.nan,
                    'Granger_P_Value': np.nan,
                    'Final_VIF': np.nan
                })

            # 3. Model Order Selection via auto_arima
            train_y_sarima = fs_train_data[target_col].dropna()
            sarima_finder = pm.auto_arima(
                train_y_sarima, seasonal=True, m=12, d=1, D=0,
                information_criterion='aic', stepwise=False, maxiter=200,
                suppress_warnings=False, trace=False, error_action='warn'
            )
            sarima_order, sarima_seasonal_order = sarima_finder.order, sarima_finder.seasonal_order

            # Effective sample size after d=1 differencing
            eff_start_sarima = train_y_sarima.index[1].strftime('%Y-%m')
            eff_end_sarima = train_y_sarima.index[-1].strftime('%Y-%m')
            eff_n_sarima = len(train_y_sarima) - 1

            info_criteria_results.append({
                'Model': 'SARIMA',
                'Year': current_date.year,
                'Effective_Training_Period': f"{eff_start_sarima} to {eff_end_sarima}",
                'Effective_N': eff_n_sarima,
                'Selected_Order': f"{sarima_order}{sarima_seasonal_order}",
                'Num_Exog_Features': 0,
                'AIC': round(sarima_finder.aic(), 2),
                'BIC': round(sarima_finder.bic(), 2)
            })

            if selected_feature_cols:
                X_sarimax_fit = X_lagged_full.loc[fs_train_data.index, selected_feature_cols].dropna()
                y_sarimax_fit = train_y_sarima.loc[X_sarimax_fit.index]

                sarimax_finder = pm.auto_arima(
                    y_sarimax_fit, X=X_sarimax_fit, seasonal=True, m=12, d=1, D=0,
                    information_criterion='aic', stepwise=False, maxiter=200,
                    suppress_warnings=False, trace=False, error_action='warn'
                )
                sarimax_order, sarimax_seasonal_order = sarimax_finder.order, sarimax_finder.seasonal_order

                eff_start_sarimax = y_sarimax_fit.index[1].strftime('%Y-%m')
                eff_end_sarimax = y_sarimax_fit.index[-1].strftime('%Y-%m')
                eff_n_sarimax = len(y_sarimax_fit) - 1

                info_criteria_results.append({
                    'Model': 'SARIMAX',
                    'Year': current_date.year,
                    'Effective_Training_Period': f"{eff_start_sarimax} to {eff_end_sarimax}",
                    'Effective_N': eff_n_sarimax,
                    'Selected_Order': f"{sarimax_order}{sarimax_seasonal_order}",
                    'Num_Exog_Features': len(selected_feature_cols),
                    'AIC': round(sarimax_finder.aic(), 2),
                    'BIC': round(sarimax_finder.bic(), 2)
                })
            else:
                sarimax_order, sarimax_seasonal_order = sarima_order, sarima_seasonal_order
                info_criteria_results.append({
                    'Model': 'SARIMAX',
                    'Year': current_date.year,
                    'Effective_Training_Period': f"{eff_start_sarima} to {eff_end_sarima} (Reduced to SARIMA)",
                    'Effective_N': eff_n_sarima,
                    'Selected_Order': f"{sarimax_order}{sarimax_seasonal_order}",
                    'Num_Exog_Features': 0,
                    'AIC': np.nan,
                    'BIC': np.nan
                })

        # ======================================================================
        # 6B. Monthly Refit & 2-Step-Ahead Forecast
        # ======================================================================
        train_y_monthly = df.loc[(df.index >= TRAINING_START_DATE) & (df.index <= train_end_date), target_col].dropna()

        # Fit & Predict SARIMA
        try:
            model_sarima = pm.ARIMA(order=sarima_order, seasonal_order=sarima_seasonal_order, maxiter=200).fit(train_y_monthly)
            if not model_sarima.arima_res_.mle_retvals.get('converged', False):
                raise RuntimeError("L-BFGS-B did not converge")
        except (LinAlgError, RuntimeError, ValueError) as e:
            print(f"  [Fallback] SARIMA switched to Nelder-Mead at {current_date.strftime('%Y-%m')} ({e})")
            model_sarima = pm.ARIMA(order=sarima_order, seasonal_order=sarima_seasonal_order, method='nm', maxiter=2000).fit(train_y_monthly)

        sarima_pred_val = model_sarima.predict(n_periods=2).iloc[1]
        predictions_sarima.append(sarima_pred_val)
        monthly_diagnostics_list.append(extract_diagnostics(model_sarima, 'SARIMA', current_date))
        monthly_internal_params_list.extend(extract_internal_params(model_sarima, 'SARIMA', current_date))

        # Fit & Predict SARIMAX
        if selected_feature_cols:
            try:
                X_to_fit = X_lagged_full.loc[train_y_monthly.index, selected_feature_cols].dropna()
                y_to_fit = train_y_monthly.loc[X_to_fit.index]
                future_X_final = X_lagged_full.loc[train_end_date + pd.DateOffset(months=1):current_date, selected_feature_cols]

                if future_X_final.isna().any().any():
                    raise ValueError(f"NaN detected in future_X_final at {current_date.strftime('%Y-%m')}")

                try:
                    model_sarimax = pm.ARIMA(order=sarimax_order, seasonal_order=sarimax_seasonal_order, maxiter=200).fit(y_to_fit, X=X_to_fit)
                    if not model_sarimax.arima_res_.mle_retvals.get('converged', False):
                        raise RuntimeError("L-BFGS-B did not converge")
                except (LinAlgError, RuntimeError, ValueError) as e:
                    print(f"  [Fallback] SARIMAX switched to Nelder-Mead at {current_date.strftime('%Y-%m')} ({e})")
                    model_sarimax = pm.ARIMA(order=sarimax_order, seasonal_order=sarimax_seasonal_order, method='nm', maxiter=2000).fit(y_to_fit, X=X_to_fit)

                sarimax_pred_val = model_sarimax.predict(n_periods=2, X=future_X_final).iloc[1]
                predictions_sarimax.append(sarimax_pred_val)
                monthly_diagnostics_list.append(extract_diagnostics(model_sarimax, 'SARIMAX', current_date))
                monthly_internal_params_list.extend(extract_internal_params(model_sarimax, 'SARIMAX', current_date, selected_feature_cols))

                # Extract parameter estimates & contributions from underlying statsmodels SARIMAXResults
                res = model_sarimax.arima_res_
                for f_col in selected_feature_cols:
                    if f_col in res.params.index:
                        beta = float(res.params[f_col])
                        se = float(res.bse[f_col]) if f_col in res.bse.index else np.nan
                        z_val = float(res.zvalues[f_col]) if f_col in res.zvalues.index else np.nan
                        p_val = float(res.pvalues[f_col]) if f_col in res.pvalues.index else np.nan

                        x_std = float(X_to_fit[f_col].std())
                        x_mean = float(X_to_fit[f_col].mean())
                        std_beta = beta * x_std

                        # Exogenous value used at the 2-step target month (current_date)
                        x_val_target = float(future_X_final.loc[current_date, f_col])
                        abs_contrib_centered = abs(beta * (x_val_target - x_mean))

                        meta = current_vars_meta.get(f_col, {})
                        monthly_exog_stats_list.append({
                            'Date': current_date.strftime('%Y-%m-%d'),
                            'Year': current_date.year,
                            'Variable': meta.get('Variable', f_col.rsplit('_lag', 1)[0]),
                            'Feature_Col': f_col,
                            'Best_Lag': meta.get('Best_Lag', np.nan),
                            'Is_Differenced': meta.get('Is_Differenced', np.nan),
                            'Coef_Beta': beta,
                            'Std_Error': se,
                            'Z_Stat': z_val,
                            'Abs_Z_Stat': abs(z_val) if pd.notna(z_val) else np.nan,
                            'P_Value': p_val,
                            'Significant_005': (p_val < 0.05) if pd.notna(p_val) else False,
                            'X_Train_SD': x_std,
                            'Standardized_Beta': std_beta,
                            'Abs_Standardized_Beta': abs(std_beta),
                            'Abs_Contribution_Centered': abs_contrib_centered
                        })
            except (Exception, LinAlgError) as e:
                print(f"[WARNING] SARIMAX failed for {current_date.strftime('%Y-%m')}: {e}. Falling back to SARIMA prediction.")
                predictions_sarimax.append(sarima_pred_val)
                monthly_diagnostics_list.append(extract_diagnostics(model_sarima, 'SARIMAX', current_date))
                monthly_internal_params_list.extend(extract_internal_params(model_sarima, 'SARIMAX', current_date))
        else:
            # Assign SARIMA prediction when 0 exogenous features were selected
            predictions_sarimax.append(sarima_pred_val)
            monthly_diagnostics_list.append(extract_diagnostics(model_sarima, 'SARIMAX', current_date))
            monthly_internal_params_list.extend(extract_internal_params(model_sarima, 'SARIMAX', current_date))

        actuals.append(y_full.loc[current_date])

    # ==========================================================================
    # --- 7. Evaluation & Export ---
    # ==========================================================================
    # 7.1 Forecast vs Actual
    results_df = pd.DataFrame({
        'Actual': actuals,
        'SARIMA_Pred': predictions_sarima,
        'SARIMAX_Pred': predictions_sarimax
    }, index=validation_index)
    results_df.index.name = 'Date'
    results_df.to_csv(os.path.join(output_folder, 'forecast_vs_actual.csv'), encoding='utf-8-sig')

    mae_sarima = mean_absolute_error(results_df['Actual'], results_df['SARIMA_Pred'])
    mae_sarimax = mean_absolute_error(results_df['Actual'], results_df['SARIMAX_Pred'])
    rmse_sarima = np.sqrt(mean_squared_error(results_df['Actual'], results_df['SARIMA_Pred']))
    rmse_sarimax = np.sqrt(mean_squared_error(results_df['Actual'], results_df['SARIMAX_Pred']))

    print(f"\n--- Overall Walk-Forward Forecast Summary ---")
    print(f"SARIMA  - MAE: {mae_sarima:.2f}, RMSE: {rmse_sarima:.2f}")
    print(f"SARIMAX - MAE: {mae_sarimax:.2f}, RMSE: {rmse_sarimax:.2f}")

    # 7.2 Model Specification & Information Criteria (Hierarchical: Model > Year)
    df_info_criteria = pd.DataFrame(info_criteria_results)
    df_info_criteria['Model'] = pd.Categorical(df_info_criteria['Model'], categories=['SARIMA', 'SARIMAX'], ordered=True)
    df_info_criteria = df_info_criteria.sort_values(['Model', 'Year']).reset_index(drop=True)
    df_info_criteria.to_csv(os.path.join(output_folder, 'model_specification_aic_bic.csv'), index=False, encoding='utf-8-sig')
    print("\n--- Model Specification & AIC/BIC Summary ---")
    print(df_info_criteria.to_string(index=False))
    df_diag = pd.DataFrame(monthly_diagnostics_list)
    df_diag.to_csv(os.path.join(output_folder, 'monthly_diagnostics_details.csv'), index=False, encoding='utf-8-sig')

    # 7.3 Annual Selected Features List
    df_annual_selected = pd.DataFrame(annual_selected_features_list)
    df_annual_selected.to_csv(os.path.join(output_folder, 'annual_selected_features.csv'), index=False, encoding='utf-8-sig')
    print(f"\n-> Saved annual selected features list to: annual_selected_features.csv")

    # 7.3b Monthly Internal Parameter Details & Summary (Table S3)
    if monthly_internal_params_list:
        df_internal_monthly = pd.DataFrame(monthly_internal_params_list)
        df_internal_monthly.to_csv(os.path.join(output_folder, 'internal_params_monthly_details.csv'), index=False, encoding='utf-8-sig')

        df_arma = df_internal_monthly[df_internal_monthly['Param_Type'] == 'AR_MA'].copy()
        val_years = sorted(list(set(validation_index.year)))
        summary_periods = [(str(y), y) for y in val_years] + [('Overall', None)]
        internal_summary_rows = []

        for model_label in ['SARIMA', 'SARIMAX']:
            df_m = df_arma[df_arma['Model'] == model_label]
            for period_name, yr in summary_periods:
                if yr is not None:
                    df_sub = df_m[df_m['Year'] == yr]
                    total_months_in_period = int(sum(validation_index.year == yr))
                else:
                    df_sub = df_m
                    total_months_in_period = len(validation_index)

                if df_sub.empty:
                    continue

                unique_params = sorted(df_sub['Parameter'].unique().tolist(), key=param_sort_key)
                for param_name in unique_params:
                    grp = df_sub[df_sub['Parameter'] == param_name]
                    years_used = ", ".join([str(y) for y in sorted(grp['Year'].unique().tolist())])
                    active_months = len(grp)
                    sig_months = int(grp['Significant_005'].sum())
                    sig_rate = (sig_months / active_months) * 100.0 if active_months > 0 else 0.0

                    internal_summary_rows.append({
                        'Model': model_label,
                        'Period': period_name,
                        'Parameter': param_name,
                        'Selected_Years': years_used,
                        'Active_Months': active_months,
                        'Total_Period_Months': total_months_in_period,
                        'Significant_Months_p005': sig_months,
                        'Significant_Rate_Pct': round(sig_rate, 1),
                        'Coef_Mean': round(float(grp['Coef'].mean()), 4),
                        'Coef_SD': round(float(grp['Coef'].std()), 4) if len(grp) > 1 else 0.0,
                        'Std_Error_Mean': round(float(grp['Std_Error'].mean()), 4),
                        'Z_Stat_Mean': round(float(grp['Z_Stat'].mean()), 2),
                        'Abs_Z_Stat_Mean': round(float(grp['Abs_Z_Stat'].mean()), 2),
                        'Abs_Z_Stat_SD': round(float(grp['Abs_Z_Stat'].std()), 2) if len(grp) > 1 else 0.0,
                        'P_Value_Min': round(float(grp['P_Value'].min()), 4),
                        'P_Value_Max': round(float(grp['P_Value'].max()), 4)
                    })

        df_internal_summary = pd.DataFrame(internal_summary_rows)
        df_internal_summary.to_csv(os.path.join(output_folder, 'internal_params_summary.csv'), index=False, encoding='utf-8-sig')
        print("\n--- Internal AR/MA Parameter Summary (Annual & Overall) ---")
        print(df_internal_summary.to_string(index=False))

    # 7.4 Monthly Exogenous Details & Annual/Overall Feature Importance Summary (Table S5)
    if monthly_exog_stats_list:
        df_monthly_exog = pd.DataFrame(monthly_exog_stats_list)
        df_monthly_exog.to_csv(os.path.join(output_folder, 'exog_monthly_refit_details.csv'), index=False, encoding='utf-8-sig')

        val_years = sorted(list(set(validation_index.year)))
        summary_periods = [(str(y), y) for y in val_years] + [('Overall', None)]
        summary_rows = []

        for period_name, yr in summary_periods:
            if yr is not None:
                df_sub = df_monthly_exog[df_monthly_exog['Year'] == yr]
                total_months_in_period = int(sum(validation_index.year == yr))
            else:
                df_sub = df_monthly_exog
                total_months_in_period = len(validation_index)

            if df_sub.empty:
                continue

            # Group by original Variable name to aggregate across years/months
            for var_name, grp in df_sub.groupby('Variable', sort=False):
                lags_used = sorted(grp['Best_Lag'].dropna().unique().astype(int).tolist())
                lags_str = ", ".join([f"Lag {l}" for l in lags_used])
                years_used = ", ".join([str(y) for y in sorted(grp['Year'].unique().tolist())])

                active_months = len(grp)
                sig_months = int(grp['Significant_005'].sum())
                sig_rate = (sig_months / active_months) * 100.0 if active_months > 0 else 0.0

                std_beta_mean = grp['Standardized_Beta'].mean()
                std_beta_sd = grp['Standardized_Beta'].std() if len(grp) > 1 else 0.0
                abs_std_beta_mean = grp['Abs_Standardized_Beta'].mean()

                abs_z_mean = grp['Abs_Z_Stat'].mean()
                abs_z_sd = grp['Abs_Z_Stat'].std() if len(grp) > 1 else 0.0

                mac_mean = grp['Abs_Contribution_Centered'].mean()
                mac_sd = grp['Abs_Contribution_Centered'].std() if len(grp) > 1 else 0.0

                summary_rows.append({
                    'Period': period_name,
                    'Variable': var_name,
                    'Selected_Lags': lags_str,
                    'Selected_Years': years_used,
                    'Active_Months': active_months,
                    'Total_Period_Months': total_months_in_period,
                    'Significant_Months_p005': sig_months,
                    'Significant_Rate_Pct': round(sig_rate, 1),
                    'Std_Beta_Mean': round(std_beta_mean, 2),
                    'Std_Beta_SD': round(std_beta_sd, 2),
                    'Abs_Std_Beta_Mean': round(abs_std_beta_mean, 2),
                    'Abs_Z_Stat_Mean': round(abs_z_mean, 2),
                    'Abs_Z_Stat_SD': round(abs_z_sd, 2),
                    'MAC_Mean': round(mac_mean, 2),
                    'MAC_SD': round(mac_sd, 2)
                })

        df_exog_summary = pd.DataFrame(summary_rows)
        # Sort within each Period by Active_Months (descending) and Abs_Z_Stat_Mean (descending)
        df_exog_summary['Period'] = pd.Categorical(
            df_exog_summary['Period'],
            categories=[str(y) for y in val_years] + ['Overall'],
            ordered=True
        )
        df_exog_summary = df_exog_summary.sort_values(
            by=['Period', 'Active_Months', 'Abs_Z_Stat_Mean'],
            ascending=[True, False, False]
        ).reset_index(drop=True)

        df_exog_summary.to_csv(os.path.join(output_folder, 'exog_feature_importance_summary.csv'), index=False, encoding='utf-8-sig')
        print("\n--- Exogenous Feature Importance Summary (Annual & Overall) ---")
        print(df_exog_summary.to_string(index=False))
    else:
        print("\n[INFO] No exogenous variables were active in any validation month. Feature importance summary skipped.")

    print(f"\n--- Walk-forward validation completed successfully. All outputs saved in: {output_folder} ---")

except FileNotFoundError:
    print(f"\n★★★ ERROR: '{file_name}' not found. ★★★")
except Exception:
    import traceback
    traceback.print_exc()
finally:
    if 'original_stdout' in locals() and isinstance(sys.stdout, Tee):
        sys.stdout = original_stdout
        log_file.close()

    if 'warnings_file' in locals() and not warnings_file.closed:
        warnings_file.close()
