# ====================================================================
# SARIMA / SARIMAX Evaluation & Statistical Analysis (HLN-DM Two-Sided)
# File: code_analyze_2.py
# ====================================================================
import pandas as pd
import numpy as np
from scipy.stats import t as student_t
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import argparse
import os
import sys
from datetime import datetime

# ====================================================================
# 基本設定 (Basic Configurations)
# ====================================================================
BASE_FOLDER_NAME = '.'
Y_MIN = None
Y_MAX = None

# HLN-DM検定の設定 (両側 alpha=0.05)
H = 2
KERNEL = 'bartlett'
ALPHA = 0.05

# 色覚多様性対応パレット (Okabe-Ito Colorblind-Safe Palette)
COLOR_ACTUAL = '#000000'   # Black
COLOR_SARIMA = '#0072B2'   # Blue
COLOR_SARIMAX = '#D55E00'  # Vermilion (Red)


# ====================================================================
# Logger Class
# ====================================================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ====================================================================
# Statistical Test Functions (HLN-Modified DM Test: Two-Sided)
# ====================================================================
def compute_loss_series(errors, loss_type='mae'):
    err = np.asarray(errors, dtype=float)
    if loss_type == 'mae':
        return np.abs(err)
    elif loss_type == 'mse':
        return err ** 2
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


def hln_dm_test(d_series, h=2, kernel='bartlett', alpha=0.05):
    d = np.asarray(d_series, dtype=float)
    d = d[~np.isnan(d)]
    n = len(d)
    df = n - 1

    if n <= h:
        return {
            'n': n, 'df': df, 'd_mean': np.nan, 'dm_stat': np.nan, 'hln_stat': np.nan,
            'p_two_sided': np.nan, 'decision_two_sided': 'NaN'
        }

    d_mean = np.mean(d)
    d_centered = d - d_mean

    gamma_0 = np.sum(d_centered ** 2) / n
    gamma_sum = 0.0
    max_lag = h - 1

    for k in range(1, max_lag + 1):
        gamma_k = np.sum(d_centered[k:] * d_centered[:-k]) / n
        weight = 1.0 - (k / (max_lag + 1.0)) if kernel == 'bartlett' else 1.0
        gamma_sum += 2.0 * weight * gamma_k

    lrv = gamma_0 + gamma_sum
    if lrv <= 0:
        lrv = gamma_0

    var_d_mean = lrv / n
    if var_d_mean <= 0:
        return {
            'n': n, 'df': df, 'd_mean': d_mean, 'dm_stat': 0.0, 'hln_stat': 0.0,
            'p_two_sided': 1.0, 'decision_two_sided': 'Not Significant'
        }

    dm_stat = d_mean / np.sqrt(var_d_mean)
    hln_factor = np.sqrt((n + 1.0 - 2.0 * h + (h * (h - 1.0)) / n) / n)
    hln_stat = dm_stat * hln_factor

    p_two_sided = 2.0 * (1.0 - student_t.cdf(np.abs(hln_stat), df=df))

    if p_two_sided < alpha and d_mean > 0:
        dec_2s = f"Sig. Improved (2S p<{alpha})"
    elif p_two_sided < alpha and d_mean < 0:
        dec_2s = f"Sig. Degraded (2S p<{alpha})"
    else:
        dec_2s = "Not Significant"

    return {
        'n': n, 'df': df, 'd_mean': d_mean, 'dm_stat': dm_stat, 'hln_stat': hln_stat,
        'p_two_sided': p_two_sided, 'decision_two_sided': dec_2s
    }


def resolve_forecast_csv(folder_path):
    candidates = [
        'forecast_vs_actual.csv',
        'forecast_vs_actual_sarimax.csv',
        'forecast_vs_actual_sarimax.scv'
    ]
    for fname in candidates:
        fpath = os.path.join(folder_path, fname)
        if os.path.exists(fpath):
            return fpath
    return None


# ====================================================================
# Main Execution Pipeline
# ====================================================================
def main():
    parser = argparse.ArgumentParser(description='SARIMA/SARIMAXの評価指標計算・HLN-DM検定・論文用プロット生成を一括実行します。')
    parser.add_argument('target_folder', type=str, help='forecast_vs_actual.csvが含まれるターゲットフォルダへのパス')
    parser.add_argument('base_file_name', type=str, help='元系列データのCSVファイル名（例: data_japan_all_get260813.csv）')
    parser.add_argument('--group_folders', nargs='*', default=None,
                        help='同一グループの複数フォルダ間で縦軸スケールを共通化する場合にフォルダパスを列挙（例: tokyo_dir tokyo_met_dir）')

    args = parser.parse_args()
    target_folder = args.target_folder

    base_file_path = os.path.join(BASE_FOLDER_NAME, args.base_file_name)
    forecast_file_path = resolve_forecast_csv(target_folder)

    if not os.path.exists(target_folder):
        print(f"Error: ターゲットフォルダが見つかりません: {target_folder}")
        sys.exit(1)
    if not os.path.exists(base_file_path):
        print(f"Error: 元データファイルが見つかりません: {base_file_path}")
        sys.exit(1)
    if forecast_file_path is None:
        print(f"Error: 予測結果ファイルが見つかりません: {target_folder}")
        sys.exit(1)

    log_path = os.path.join(target_folder, 'analysis_report_log.txt')
    sys.stdout = Logger(log_path)

    print("===================================================================")
    print("SARIMA / SARIMAX Evaluation & HLN-DM Pipeline (Two-Sided alpha=0.05)")
    print(f"Execution Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Folder       : {os.path.abspath(target_folder)}")
    print(f"Reading forecast    : {forecast_file_path}")
    print(f"Reading base data   : {base_file_path}")
    print("===================================================================\n")

    # ====================================================================
    # 1. 評価指標マトリクス（MAE, RMSE, MASE）の計算
    # ====================================================================
    df_base = pd.read_csv(base_file_path, index_col=0, parse_dates=True)
    target_col_name = df_base.columns[0]

    train_actual = df_base.loc['2009-01-01':'2021-12-31', target_col_name]
    seasonal_naive_errors = np.abs(train_actual.diff(periods=12).dropna())
    fixed_scaling_factor = seasonal_naive_errors.mean()

    print("--- MASE Scaling Factor (Denominator) ---")
    print(f" Target Variable : {target_col_name}")
    print(f" Training Period : 2009-2021 (Seasonal Naive, m=12)")
    print(f" Scaling Factor  : {fixed_scaling_factor:.4f}\n")

    df_forecast = pd.read_csv(forecast_file_path, index_col=0, parse_dates=True)
    actual = df_forecast['Actual']
    pred_sarima = df_forecast['SARIMA_Pred']
    pred_sarimax = df_forecast['SARIMAX_Pred']

    years = sorted(list(set(df_forecast.index.year)))
    periods = [(str(y), y) for y in years] + [('Overall', None)]
    results = []

    for period_name, year in periods:
        if year is not None:
            mask = df_forecast.index.year == year
        else:
            mask = np.ones(len(df_forecast), dtype=bool)

        a = actual[mask]
        p_sarima = pred_sarima[mask]
        p_sarimax = pred_sarimax[mask]

        mae_sarima = np.mean(np.abs(a - p_sarima))
        rmse_sarima = np.sqrt(np.mean((a - p_sarima)**2))
        mase_sarima = mae_sarima / fixed_scaling_factor

        mae_sarimax = np.mean(np.abs(a - p_sarimax))
        rmse_sarimax = np.sqrt(np.mean((a - p_sarimax)**2))
        mase_sarimax = mae_sarimax / fixed_scaling_factor

        results.append({
            'Period': period_name,
            'Model': 'SARIMA',
            'MAE': round(mae_sarima, 2),
            'RMSE': round(rmse_sarima, 2),
            'MASE': round(mase_sarima, 3)
        })
        results.append({
            'Period': period_name,
            'Model': 'SARIMAX',
            'MAE': round(mae_sarimax, 2),
            'RMSE': round(rmse_sarimax, 2),
            'MASE': round(mase_sarimax, 3)
        })

    metrics_df = pd.DataFrame(results)
    print("--- Period Performance Matrix (Supplement) ---")
    print(metrics_df.to_string(index=False))
    print("-" * 45 + "\n")

    output_csv_period = os.path.join(target_folder, 'period_performance_matrix.csv')
    metrics_df.to_csv(output_csv_period, index=False, encoding='utf-8-sig')
    print(f"-> Saved period metrics matrix (Supplement) to: {output_csv_period}")

    df_overall = metrics_df[metrics_df['Period'] == 'Overall'].drop(columns=['Period']).copy()
    print("\n--- Final Performance Matrix (Overall - Main Text) ---")
    print(df_overall.to_string(index=False))

    output_csv_final = os.path.join(target_folder, 'final_performance_matrix.csv')
    df_overall.to_csv(output_csv_final, index=False, encoding='utf-8-sig')
    print(f"-> Saved overall final metrics (Main Text) to: {output_csv_final}\n")

    # ====================================================================
    # 2. HLN-Modified Diebold-Mariano Test (SARIMAX vs SARIMA, Two-Sided)
    # ====================================================================
    err_sarima = actual - pred_sarima
    err_sarimax = actual - pred_sarimax

    hln_records = []
    for loss_type in ['mae', 'mse']:
        loss_base_full = pd.Series(compute_loss_series(err_sarima, loss_type=loss_type), index=df_forecast.index)
        loss_adv_full = pd.Series(compute_loss_series(err_sarimax, loss_type=loss_type), index=df_forecast.index)
        d_full = loss_base_full - loss_adv_full

        for period_name, year in periods:
            if year is not None:
                d_sub = d_full[d_full.index.year == year]
            else:
                d_sub = d_full

            res = hln_dm_test(d_sub, h=H, kernel=KERNEL, alpha=ALPHA)
            hln_records.append({
                'Period': period_name,
                'Comparison': 'SARIMAX vs SARIMA',
                'Loss_Type': loss_type.upper(),
                'n': res['n'],
                'df': res['df'],
                'Mean_Loss_Diff': round(res['d_mean'], 4) if pd.notna(res['d_mean']) else np.nan,
                'DM_Stat': round(res['dm_stat'], 4) if pd.notna(res['dm_stat']) else np.nan,
                'HLN_DM_Stat': round(res['hln_stat'], 4) if pd.notna(res['hln_stat']) else np.nan,
                'p_2S': round(res['p_two_sided'], 4) if pd.notna(res['p_two_sided']) else np.nan,
                'Decision_2S': res['decision_two_sided']
            })

    df_hln = pd.DataFrame(hln_records)
    hln_csv_path = os.path.join(target_folder, 'period_hln_dm_test_summary.csv')
    df_hln.to_csv(hln_csv_path, index=False, encoding='utf-8-sig')

    print("--- HLN-DM Test Summary (SARIMAX vs SARIMA, Two-Sided alpha=0.05) ---")
    cols_show = ['Period', 'Loss_Type', 'Mean_Loss_Diff', 'HLN_DM_Stat', 'p_2S', 'Decision_2S']
    print(df_hln[cols_show].to_string(index=False))
    print(f"-> Saved HLN-DM test results to: {hln_csv_path}\n")

    # ====================================================================
    # 2.5 モデル仕様・AIC/BIC・月次残差診断の統合サマリー (Supplement Table S2)
    # ====================================================================
    spec_file_path = os.path.join(target_folder, 'model_specification_aic_bic.csv')
    diag_file_path = os.path.join(target_folder, 'monthly_diagnostics_details.csv')

    if os.path.exists(spec_file_path) and os.path.exists(diag_file_path):
        df_spec = pd.read_csv(spec_file_path)
        df_diag = pd.read_csv(diag_file_path)

        if 'Effective_Training_Period' in df_spec.columns:
            period_split = df_spec['Effective_Training_Period'].astype(str).str.split(' to ', n=1, expand=True)
            if period_split.shape[1] == 2:
                df_spec['Train_Start'] = period_split[0]
                df_spec['Train_End'] = period_split[1].str.replace(' (Reduced to SARIMA)', '', regex=False)
            else:
                df_spec['Train_Start'] = df_spec['Effective_Training_Period']
                df_spec['Train_End'] = np.nan

        df_spec = df_spec.rename(columns={'AIC': 'AIC_Jan', 'BIC': 'BIC_Jan'})

        diag_summary_list = []
        for (model_name, yr), grp in df_diag.groupby(['Model', 'Year'], sort=False):
            diag_summary_list.append({
                'Model': model_name,
                'Year': int(yr),
                'LB_Lag1_P_Min': round(float(grp['LB_P_Lag1'].min()), 4),
                'LB_Lag1_P_Max': round(float(grp['LB_P_Lag1'].max()), 4),
                'LB_Lag1_Pass_Months': int(grp['LB_Pass_Lag1'].sum()),
                'LB_Lag12_P_Min': round(float(grp['LB_P_Lag12'].min()), 4),
                'LB_Lag12_P_Max': round(float(grp['LB_P_Lag12'].max()), 4),
                'LB_Lag12_Pass_Months': int(grp['LB_Pass_Lag12'].sum()),
                'JB_P_Min': round(float(grp['JB_P'].min()), 4),
                'JB_P_Max': round(float(grp['JB_P'].max()), 4),
                'JB_Pass_Months': int(grp['JB_Pass'].sum())
            })

        df_diag_sum = pd.DataFrame(diag_summary_list)
        df_supp_diag = pd.merge(df_spec, df_diag_sum, on=['Model', 'Year'], how='left')

        preferred_cols = [
            'Model', 'Year', 'Train_Start', 'Train_End', 'Effective_N',
            'Selected_Order', 'Num_Exog_Features', 'AIC_Jan', 'BIC_Jan',
            'LB_Lag1_P_Min', 'LB_Lag1_P_Max', 'LB_Lag1_Pass_Months',
            'LB_Lag12_P_Min', 'LB_Lag12_P_Max', 'LB_Lag12_Pass_Months',
            'JB_P_Min', 'JB_P_Max', 'JB_Pass_Months'
        ]
        final_cols = [c for c in preferred_cols if c in df_supp_diag.columns]
        df_supp_diag = df_supp_diag[final_cols]

        supp_num_path = os.path.join(target_folder, 'supplement_model_diagnostics_numeric.csv')
        df_supp_diag.to_csv(supp_num_path, index=False, encoding='utf-8-sig')

        df_supp_formatted = df_supp_diag.copy()
        p_cols = ['LB_Lag1_P_Min', 'LB_Lag1_P_Max', 'LB_Lag12_P_Min', 'LB_Lag12_P_Max', 'JB_P_Min', 'JB_P_Max']
        for col in p_cols:
            if col in df_supp_formatted.columns:
                df_supp_formatted[col] = df_supp_formatted[col].apply(
                    lambda x: '<.0001' if pd.notna(x) and x < 0.0001 else (f"{x:.4f}" if pd.notna(x) else "")
                )

        supp_fmt_path = os.path.join(target_folder, 'supplement_model_diagnostics_formatted.csv')
        df_supp_formatted.to_csv(supp_fmt_path, index=False, encoding='utf-8-sig')

        print("--- Supplement Model Specification & Diagnostics Summary (Table S2) ---")
        print(df_supp_formatted.to_string(index=False))
        print(f"-> Saved numeric supplement table to   : {supp_num_path}")
        print(f"-> Saved formatted supplement table to : {supp_fmt_path}\n")
    else:
        print("[INFO] Specification or monthly diagnostics CSV not found. Skipping supplement diagnostics table.\n")

    # ====================================================================
    # 2.6 内部パラメータ (AR/MA項) の統計的有意性・Z統計量サマリー (Supplement Table S3)
    # ====================================================================
    internal_sum_path = os.path.join(target_folder, 'internal_params_summary.csv')

    if os.path.exists(internal_sum_path):
        df_s3_num = pd.read_csv(internal_sum_path)
        s3_num_out = os.path.join(target_folder, 'supplement_table_s3_internal_params_numeric.csv')
        df_s3_num.to_csv(s3_num_out, index=False, encoding='utf-8-sig')

        df_s3_fmt = df_s3_num.copy()
        for pcol in ['P_Value_Min', 'P_Value_Max']:
            if pcol in df_s3_fmt.columns:
                df_s3_fmt[pcol] = df_s3_fmt[pcol].apply(
                    lambda x: '<.0001' if pd.notna(x) and x < 0.0001 else (f"{x:.4f}" if pd.notna(x) else "")
                )

        s3_fmt_out = os.path.join(target_folder, 'supplement_table_s3_internal_params_formatted.csv')
        df_s3_fmt.to_csv(s3_fmt_out, index=False, encoding='utf-8-sig')

        print("--- Supplement Internal AR/MA Parameter Summary (Table S3) ---")
        print(df_s3_fmt.to_string(index=False))
        print(f"-> Saved numeric Table S3 to   : {s3_num_out}")
        print(f"-> Saved formatted Table S3 to : {s3_fmt_out}\n")
    else:
        print("[INFO] internal_params_summary.csv not found. Skipping Table S3.\n")

    # ====================================================================
    # 3. 縦軸スケールの動的決定と論文用プロットの生成 (Figure 2)
    # ====================================================================
    group_dfs = [df_forecast]
    if args.group_folders:
        for g_folder in args.group_folders:
            g_csv = resolve_forecast_csv(g_folder)
            if g_csv and os.path.exists(g_csv):
                group_dfs.append(pd.read_csv(g_csv, index_col=0, parse_dates=True))

    all_vals_list = []
    for g_df in group_dfs:
        for col in ['Actual', 'SARIMA_Pred', 'SARIMAX_Pred']:
            if col in g_df.columns:
                all_vals_list.append(g_df[col].dropna().values)

    all_group_vals = np.concatenate(all_vals_list)
    min_val, max_val = float(np.nanmin(all_group_vals)), float(np.nanmax(all_group_vals))
    margin_val = (max_val - min_val) * 0.05 if max_val != min_val else 10.0

    y_min_plot = Y_MIN if Y_MIN is not None else (min_val - margin_val)
    y_max_plot = Y_MAX if Y_MAX is not None else (max_val + margin_val)

    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.spines.top': True,
        'axes.spines.right': True
    })

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    ax.plot(df_forecast.index, df_forecast['Actual'], label='Actual',
            color=COLOR_ACTUAL, linestyle='-', marker='o', markersize=5, linewidth=1.8, zorder=3)
    ax.plot(df_forecast.index, df_forecast['SARIMA_Pred'], label='SARIMA',
            color=COLOR_SARIMA, linestyle='--', linewidth=2.0, zorder=2)
    ax.plot(df_forecast.index, df_forecast['SARIMAX_Pred'], label='SARIMAX',
            color=COLOR_SARIMAX, linestyle='-.', linewidth=2.0, zorder=1)

    ax.set_xlabel('Month (Jan 2022 – Dec 2024)')
    ax.set_ylabel('Monthly suicide counts')
    ax.grid(False)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.xticks(rotation=45, ha='right')

    start_date = df_forecast.index.min() - pd.Timedelta(days=20)
    end_date = df_forecast.index.max() + pd.Timedelta(days=20)
    ax.set_xlim(start_date, end_date)
    ax.set_ylim(y_min_plot, y_max_plot)
    print(f"-> Y-axis range dynamically set to: {y_min_plot:.2f} - {y_max_plot:.2f}")

    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()

    plot_out_path = os.path.join(target_folder, 'publication_forecast_plot.png')
    fig.savefig(plot_out_path, dpi=300)
    plt.close(fig)

    print(f"-> Saved plot to: {plot_out_path}")
    print(f"-> Saved log report to: {log_path}\n")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
