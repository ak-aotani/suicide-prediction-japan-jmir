# ===================================================================
# LightGBM 10-Seed Ensemble & Statistical Evaluation (HLN-DM)
# File: code_analyze_10seed.py
# ===================================================================
import pandas as pd
import numpy as np
from scipy.stats import t as student_t
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import argparse
import glob
from datetime import datetime

# 色覚多様性対応パレット (Okabe-Ito Colorblind-Safe Palette)
COLOR_ACTUAL = '#000000'    # Black (Actual)
COLOR_INTERNAL = '#0072B2'  # Blue (Internal-feature-only: Models A & B)
COLOR_EXTERNAL = '#D55E00'  # Vermilion / Red (External-variable: Models C & D)
COLOR_GREEN = '#009E73'     # Bluish Green
BAND_ALPHA = 0.18           # Min-Max band transparency


# ===================================================================
# Logger Class
# ===================================================================
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


# ===================================================================
# 1. Statistical Test Functions (HLN-Modified DM Test: Two-Sided)
# ===================================================================
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


def jaccard_index(set1, set2):
    if not set1 and not set2:
        return 1.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


# ===================================================================
# 2. Pairwise Evaluation Function (Period-aware, Two-Sided alpha=0.05)
# ===================================================================
def evaluate_lgbm_pair(df_preds_all, base_model, adv_model, seeds, period_name, year, h=2, loss_type='mae', kernel='bartlett', alpha=0.05):
    seed_records = []
    d_matrix = []

    if year is not None:
        df_filtered = df_preds_all[df_preds_all.index.year == year]
    else:
        df_filtered = df_preds_all

    for seed in seeds:
        df_s = df_filtered[df_filtered['Seed'] == seed].sort_index()
        if df_s.empty:
            continue

        loss_base = compute_loss_series(df_s[f'Error_{base_model}'], loss_type=loss_type)
        loss_adv = compute_loss_series(df_s[f'Error_{adv_model}'], loss_type=loss_type)
        d_seed = pd.Series(loss_base - loss_adv, index=df_s.index)
        d_matrix.append(d_seed)

        res_seed = hln_dm_test(d_seed, h=h, kernel=kernel, alpha=alpha)
        seed_records.append({
            'Seed': seed,
            'Sig_Improve_2S': (res_seed['p_two_sided'] < alpha) and (res_seed['d_mean'] > 0),
            'Sig_Degrade_2S': (res_seed['p_two_sided'] < alpha) and (res_seed['d_mean'] < 0)
        })

    if not d_matrix:
        return None
    df_seed_res = pd.DataFrame(seed_records)

    d_bar_series = pd.concat(d_matrix, axis=1).mean(axis=1)
    res_agg = hln_dm_test(d_bar_series, h=h, kernel=kernel, alpha=alpha)

    summary_record = {
        'Period': period_name,
        'Comparison': f'{adv_model} vs {base_model}',
        'Loss_Type': loss_type.upper(),
        'n': res_agg['n'],
        'df': res_agg['df'],
        'Mean_Loss_Diff': round(res_agg['d_mean'], 4),
        'HLN_DM_Stat': round(res_agg['hln_stat'], 4),
        'p_2S': round(res_agg['p_two_sided'], 4),
        'Decision_2S': res_agg['decision_two_sided'],
        'Seeds_Sig_Improve_2S': f"{int(df_seed_res['Sig_Improve_2S'].sum())} out of {len(seeds)}",
        'Seeds_Sig_Degrade_2S': f"{int(df_seed_res['Sig_Degrade_2S'].sum())} out of {len(seeds)}"
    }
    return summary_record


# ===================================================================
# 3. Publication & Diagnostic Plots Generator
# ===================================================================
def apply_common_time_axis(ax, dates, y_lim):
    ax.set_xlabel('Month (Jan 2022 – Dec 2024)')
    ax.set_ylabel('Monthly suicide counts')
    ax.grid(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')
    start_date = dates.min() - pd.Timedelta(days=20)
    end_date = dates.max() + pd.Timedelta(days=20)
    ax.set_xlim(start_date, end_date)
    ax.set_ylim(y_lim)


def generate_publication_figure3_plots(df_preds_all, df_ensemble_pred, output_dir, y_lim_preds):
    """
    Figure 3 用のペア比較プロット (Model A vs Model C, Model B vs Model D) を
    10シード平均線 + Min-Max帯、Okabe-Ito配色、共通縦軸・横軸スケールで出力します。
    """
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.spines.top': True,
        'axes.spines.right': True
    })

    dates = df_ensemble_pred.index
    first_seed = df_preds_all['Seed'].iloc[0]
    actual = df_preds_all[df_preds_all['Seed'] == first_seed].sort_index()['Actual']

    fig3_pairs = [
        ('Model_A', 'Model_C', 'Model A', 'Model C'),
        ('Model_B', 'Model_D', 'Model B', 'Model D')
    ]

    # 1) 個別パネルプロット (Model A vs C, Model B vs D)
    for int_model, ext_model, int_label, ext_label in fig3_pairs:
        piv_int = df_preds_all.reset_index().pivot(index='Date', columns='Seed', values=f'Pred_{int_model}').loc[dates]
        piv_ext = df_preds_all.reset_index().pivot(index='Date', columns='Seed', values=f'Pred_{ext_model}').loc[dates]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

        h_act, = ax.plot(dates, actual, color=COLOR_ACTUAL, linestyle='-', marker='o',
                         markersize=5, linewidth=1.8, zorder=5)

        h_int_line, = ax.plot(dates, piv_int.mean(axis=1), color=COLOR_INTERNAL,
                              linestyle='--', linewidth=2.0, zorder=4)
        h_int_band = ax.fill_between(dates, piv_int.min(axis=1), piv_int.max(axis=1),
                                     color=COLOR_INTERNAL, alpha=BAND_ALPHA, edgecolor='none', zorder=2)

        h_ext_line, = ax.plot(dates, piv_ext.mean(axis=1), color=COLOR_EXTERNAL,
                              linestyle='-.', linewidth=2.0, zorder=3)
        h_ext_band = ax.fill_between(dates, piv_ext.min(axis=1), piv_ext.max(axis=1),
                                     color=COLOR_EXTERNAL, alpha=BAND_ALPHA, edgecolor='none', zorder=1)

        apply_common_time_axis(ax, dates, y_lim_preds)

        ax.legend(
            handles=[h_act, (h_int_band, h_int_line), (h_ext_band, h_ext_line)],
            labels=['Actual', f'{int_label} (Mean & Min–Max)', f'{ext_label} (Mean & Min–Max)'],
            loc='upper right',
            frameon=True
        )
        plt.tight_layout()
        out_path = os.path.join(output_dir, f'publication_forecast_plot_{int_model}_vs_{ext_model}.png')
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    # 2) 左右2パネル結合プロット (左: Model A vs C, 右: Model B vs D)
    fig, axes = plt.subplots(1, 2, figsize=(18, 5), dpi=300, sharey=True)
    for idx, (int_model, ext_model, int_label, ext_label) in enumerate(fig3_pairs):
        ax = axes[idx]
        piv_int = df_preds_all.reset_index().pivot(index='Date', columns='Seed', values=f'Pred_{int_model}').loc[dates]
        piv_ext = df_preds_all.reset_index().pivot(index='Date', columns='Seed', values=f'Pred_{ext_model}').loc[dates]

        h_act, = ax.plot(dates, actual, color=COLOR_ACTUAL, linestyle='-', marker='o',
                         markersize=5, linewidth=1.8, zorder=5)
        h_int_line, = ax.plot(dates, piv_int.mean(axis=1), color=COLOR_INTERNAL,
                              linestyle='--', linewidth=2.0, zorder=4)
        h_int_band = ax.fill_between(dates, piv_int.min(axis=1), piv_int.max(axis=1),
                                     color=COLOR_INTERNAL, alpha=BAND_ALPHA, edgecolor='none', zorder=2)
        h_ext_line, = ax.plot(dates, piv_ext.mean(axis=1), color=COLOR_EXTERNAL,
                              linestyle='-.', linewidth=2.0, zorder=3)
        h_ext_band = ax.fill_between(dates, piv_ext.min(axis=1), piv_ext.max(axis=1),
                                     color=COLOR_EXTERNAL, alpha=BAND_ALPHA, edgecolor='none', zorder=1)

        apply_common_time_axis(ax, dates, y_lim_preds)
        ax.tick_params(labelleft=True)
        ax.legend(
            handles=[h_act, (h_int_band, h_int_line), (h_ext_band, h_ext_line)],
            labels=['Actual', f'{int_label} (Mean & Min–Max)', f'{ext_label} (Mean & Min–Max)'],
            loc='upper right',
            frameon=True
        )

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'publication_figure3_row_AC_BD.png'), dpi=300)
    plt.close(fig)


def generate_prediction_bands(df_preds_all, df_ensemble_pred, target_model, output_dir, y_lim_preds):
    dates = df_ensemble_pred.index
    preds_model = df_preds_all.reset_index().pivot(index='Date', columns='Seed', values=f'Pred_{target_model}').loc[dates]
    actual = df_preds_all[df_preds_all['Seed'] == df_preds_all['Seed'].iloc[0]].sort_index()['Actual']

    is_internal = target_model in ['Model_A', 'Model_B']
    model_color = COLOR_INTERNAL if is_internal else COLOR_EXTERNAL
    model_ls = '--' if is_internal else '-.'
    clean_name = target_model.replace('_', ' ')

    # Min-Max Plot (Unified Style)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    h_act, = ax.plot(dates, actual, color=COLOR_ACTUAL, linestyle='-', marker='o', markersize=5, linewidth=1.8, zorder=4)
    h_line, = ax.plot(dates, preds_model.mean(axis=1), color=model_color, linestyle=model_ls, linewidth=2.0, zorder=3)
    h_band = ax.fill_between(dates, preds_model.min(axis=1), preds_model.max(axis=1),
                             color=model_color, alpha=BAND_ALPHA, edgecolor='none', zorder=2)

    apply_common_time_axis(ax, dates, y_lim_preds)
    ax.legend(
        handles=[h_act, (h_band, h_line)],
        labels=['Actual', f'{clean_name} (Mean & Min–Max)'],
        loc='upper right',
        frameon=True
    )
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'prediction_bands_minmax_{target_model}.png'), dpi=300)
    plt.close(fig)


def generate_diagnostic_plots(df_preds_all, df_ensemble_pred, base_model, adv_model, output_dir, y_lim_diff, y_lim_cum):
    os.makedirs(output_dir, exist_ok=True)

    d_bar = df_ensemble_pred[f'Mean_L1_Loss_{base_model}'] - df_ensemble_pred[f'Mean_L1_Loss_{adv_model}']
    win_adv = (d_bar > 0).sum()
    win_base = (d_bar < 0).sum()
    tie = (d_bar == 0).sum()
    skewness = stats.skew(d_bar.dropna())
    rho_1 = d_bar.autocorr(lag=1)

    metrics_dict = {
        'Comparison': f'{adv_model} vs {base_model}',
        'Win_Adv': win_adv,
        'Win_Base': win_base,
        'Tie': tie,
        'Win_Rate_Adv': f"{win_adv/len(d_bar):.1%}",
        'Win_Rate_Base': f"{win_base/len(d_bar):.1%}",
        'Mean_Loss_Diff': round(d_bar.mean(), 4),
        'Max_Loss_Diff': round(d_bar.max(), 4),
        'Min_Loss_Diff': round(d_bar.min(), 4),
        'Skewness': round(skewness, 4),
        'Lag1_Autocorr': round(rho_1, 4)
    }

    dates = df_ensemble_pred.index
    start_date = dates.min() - pd.Timedelta(days=20)
    end_date = dates.max() + pd.Timedelta(days=20)

    # Monthly Loss Diff & Cumulative (Bar + Line)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=300, sharex=True)
    colors = [COLOR_INTERNAL if val > 0 else COLOR_EXTERNAL for val in d_bar]
    axes[0].bar(dates, d_bar.values, width=20, color=colors)
    axes[0].axhline(0, color=COLOR_ACTUAL, linewidth=1)
    axes[0].set_title(f'Monthly Loss Differential (Blue: {adv_model} Wins, Red: {base_model} Wins)')
    axes[0].set_ylabel('Difference in MAE')
    axes[0].set_ylim(y_lim_diff)
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].plot(dates, d_bar.cumsum(), marker='o', color=COLOR_INTERNAL, linewidth=2)
    axes[1].axhline(0, color=COLOR_ACTUAL, linewidth=1, linestyle='--')
    axes[1].set_title(r'Cumulative Mean Loss Differential ($\sum \overline{d}_t$)')
    axes[1].set_xlabel('Month (Jan 2022 – Dec 2024)')
    axes[1].set_ylabel('Cumulative MAE Difference')
    axes[1].set_ylim(y_lim_cum)
    axes[1].set_xlim(start_date, end_date)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    for label in axes[1].get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'loss_differential_mean_{adv_model}_vs_{base_model}.png'), dpi=300)
    plt.close(fig)

    # Spaghetti Plot
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    preds_adv = df_preds_all.reset_index().pivot(index='Date', columns='Seed', values=f'Pred_{adv_model}')
    seeds = preds_adv.columns

    for seed in seeds:
        df_s = df_preds_all[df_preds_all['Seed'] == seed].sort_index()
        loss_base = np.abs(df_s[f'Error_{base_model}'])
        loss_adv = np.abs(df_s[f'Error_{adv_model}'])
        d_tk = loss_base - loss_adv
        ax.plot(dates, d_tk.cumsum(), color='#888888', alpha=0.45, linewidth=1.0)

    ax.plot(dates, d_bar.cumsum(), color=COLOR_EXTERNAL, linewidth=2.8, label='10-Seed Mean Cumulative Difference')
    ax.axhline(0, color=COLOR_ACTUAL, linewidth=1, linestyle='--')
    ax.set_xlabel('Month (Jan 2022 – Dec 2024)')
    ax.set_ylabel('Cumulative MAE Difference')
    ax.set_xlim(start_date, end_date)
    ax.set_ylim(y_lim_cum)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'spaghetti_cumulative_loss_{adv_model}_vs_{base_model}.png'), dpi=300)
    plt.close(fig)

    return metrics_dict


def load_group_pred_values(folder_path):
    vals = []
    seed_dirs = sorted([d for d in glob.glob(os.path.join(folder_path, 'seed_*')) if os.path.isdir(d)])
    for sdir in seed_dirs:
        pfile = os.path.join(sdir, 'all_models_forecast_comparison.csv')
        if os.path.exists(pfile):
            df_p = pd.read_csv(pfile)
            pcols = [c for c in df_p.columns if c.startswith('Pred_') or c == 'Actual']
            vals.append(df_p[pcols].values.flatten())
    for sname in ['forecast_vs_actual_sarimax.csv', 'forecast_vs_actual.csv', 'forecast_vs_actual_sarimax.scv']:
        spath = os.path.join(folder_path, sname)
        if os.path.exists(spath):
            df_s = pd.read_csv(spath)
            scols = [c for c in ['Actual', 'SARIMA_Pred', 'SARIMAX_Pred'] if c in df_s.columns]
            if scols:
                vals.append(df_s[scols].values.flatten())
    return np.concatenate(vals) if vals else np.array([])


# ===================================================================
# 4. Main Execution Pipeline
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Run LightGBM 10-Seed HLN-DM and Diagnostic analysis.")
    parser.add_argument("target_dir", type=str, help="Path to the master output directory containing seed_* folders.")
    parser.add_argument("raw_data_file", type=str, help="Path to the original master CSV file (required for MASE).")
    parser.add_argument("--out_folder", type=str, default="ensemble_summary_hln_dm", help="Output subfolder name.")
    parser.add_argument("--group_dirs", nargs='*', default=None,
                        help="Optional list of directories in the same group to unify Y-axis limits across folders.")
    args = parser.parse_args()

    TARGET_DIR = os.path.abspath(args.target_dir)
    RAW_FILE = os.path.abspath(args.raw_data_file)
    H = 2
    KERNEL = "bartlett"
    TARGET_COLUMN = 'suicide'
    VALIDATION_START = '2022-01-01'

    if not os.path.isdir(TARGET_DIR) or not os.path.exists(RAW_FILE):
        print("Error: Specified directory or raw file does not exist.")
        return

    SUMMARY_DIR = os.path.join(TARGET_DIR, args.out_folder)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    log_path = os.path.join(SUMMARY_DIR, 'analysis_report_log.txt')
    sys.stdout = Logger(log_path)

    print(f"===================================================================")
    print(f"HLN-DM & Diagnostic Evaluation Pipeline (Two-Sided alpha=0.05)")
    print(f"Execution Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target Directory    : {TARGET_DIR}")
    print(f"===================================================================\n")

    df_raw = pd.read_csv(RAW_FILE, parse_dates=[0], index_col=0)
    df_train = df_raw.loc[df_raw.index < VALIDATION_START]
    naive_mae_insample = df_train[TARGET_COLUMN].diff(12).abs().mean()
    print(f"[MASE Setup] In-sample Seasonal Naive (m=12) MAE (< {VALIDATION_START}): {naive_mae_insample:.4f}")

    seed_dirs = sorted([d for d in glob.glob(os.path.join(TARGET_DIR, 'seed_*')) if os.path.isdir(d)])
    all_preds, all_logs = [], []
    for seed_dir in seed_dirs:
        seed_num = int(os.path.basename(seed_dir).split('_')[1])
        pred_file = os.path.join(seed_dir, 'all_models_forecast_comparison.csv')
        log_file = os.path.join(seed_dir, 'walk_forward_detailed_log_all.csv')

        if os.path.exists(pred_file) and os.path.exists(log_file):
            df_pred = pd.read_csv(pred_file, parse_dates=['Date'], index_col='Date')
            df_pred['Seed'] = seed_num
            all_preds.append(df_pred)
            df_log = pd.read_csv(log_file, parse_dates=['Date'])
            df_log['Seed'] = seed_num
            all_logs.append(df_log)

    df_preds_all = pd.concat(all_preds)
    df_logs_all = pd.concat(all_logs)

    models = [m.replace('Pred_', '') for m in df_preds_all.columns if m.startswith('Pred_')]
    seeds = sorted(df_preds_all['Seed'].unique())
    dates = sorted(df_preds_all.index.unique())

    print(f"Loaded {len(seeds)} seeds: {seeds}")

    # ---------------------------------------------------------
    # 1. 精度指標の計算 (Period: 2022, 2023, 2024, Overall)
    # ---------------------------------------------------------
    years = sorted(df_preds_all.index.year.unique())
    periods = [(str(y), y) for y in years] + [('Overall', None)]

    seed_period_metrics = []
    for seed in seeds:
        df_seed = df_preds_all[df_preds_all['Seed'] == seed]
        for period_name, year in periods:
            mask = (df_seed.index.year == year) if year is not None else np.ones(len(df_seed), dtype=bool)
            df_subset = df_seed[mask]
            if df_subset.empty:
                continue
            row = {'Seed': seed, 'Period': period_name}
            for model in models:
                mae = df_subset[f'Error_{model}'].abs().mean()
                rmse = np.sqrt((df_subset[f'Error_{model}'] ** 2).mean())
                mase = mae / naive_mae_insample
                row[f'MAE_{model}'] = mae
                row[f'RMSE_{model}'] = rmse
                row[f'MASE_{model}'] = mase
            seed_period_metrics.append(row)

    df_seed_period = pd.DataFrame(seed_period_metrics)
    df_seed_period.to_csv(os.path.join(SUMMARY_DIR, 'seed_metrics_raw.csv'), index=False, encoding='utf-8-sig')

    final_metrics_rows = []
    for period_name, _ in periods:
        df_p = df_seed_period[df_seed_period['Period'] == period_name]
        for model in models:
            final_metrics_rows.append({
                'Period': period_name,
                'Model': model,
                'MAE_Mean': round(df_p[f'MAE_{model}'].mean(), 2),
                'MAE_SD': round(df_p[f'MAE_{model}'].std(), 2),
                'RMSE_Mean': round(df_p[f'RMSE_{model}'].mean(), 2),
                'RMSE_SD': round(df_p[f'RMSE_{model}'].std(), 2),
                'MASE_Mean': round(df_p[f'MASE_{model}'].mean(), 3),
                'MASE_SD': round(df_p[f'MASE_{model}'].std(), 3)
            })

    df_period_matrix = pd.DataFrame(final_metrics_rows)
    df_period_matrix.to_csv(os.path.join(SUMMARY_DIR, 'period_performance_matrix.csv'), index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------
    # 2. 比較ペアのHLN-DM検定 (5パターン)
    # ---------------------------------------------------------
    comparison_pairs = [
        ('Model_A', 'Model_B'),
        ('Model_A', 'Model_C'),
        ('Model_A', 'Model_D'),
        ('Model_B', 'Model_D'),
        ('Model_C', 'Model_D')
    ]

    all_summary_tests = []
    for loss_type in ['mae', 'mse']:
        for base_m, adv_m in comparison_pairs:
            for period_name, year in periods:
                summary_rec = evaluate_lgbm_pair(
                    df_preds_all, base_model=base_m, adv_model=adv_m,
                    seeds=seeds, period_name=period_name, year=year, h=H, loss_type=loss_type, kernel=KERNEL, alpha=0.05
                )
                if summary_rec:
                    all_summary_tests.append(summary_rec)

    df_hln_summary = pd.DataFrame(all_summary_tests)

    period_order = [p[0] for p in periods]
    df_hln_summary['Period'] = pd.Categorical(df_hln_summary['Period'], categories=period_order, ordered=True)

    pair_order = [f"{adv} vs {base}" for base, adv in comparison_pairs]
    df_hln_summary['Comparison'] = pd.Categorical(df_hln_summary['Comparison'], categories=pair_order, ordered=True)

    df_hln_summary = df_hln_summary.sort_values(['Loss_Type', 'Period', 'Comparison'])
    df_hln_summary.to_csv(os.path.join(SUMMARY_DIR, 'period_hln_dm_test_summary.csv'), index=False, encoding='utf-8-sig')

    print("\n[HLN-DM Test Summary - Period Layout (MAE, Two-Sided alpha=0.05)]")
    df_mae = df_hln_summary[df_hln_summary['Loss_Type'] == 'MAE'].copy()
    cols_show = [
        'Period', 'Comparison', 'Mean_Loss_Diff', 'HLN_DM_Stat',
        'p_2S', 'Decision_2S', 'Seeds_Sig_Improve_2S', 'Seeds_Sig_Degrade_2S'
    ]
    print(df_mae[cols_show].to_string(index=False))

    # ---------------------------------------------------------
    # 3. Y軸動的計算とプロット生成 (Figure 3 ペアプロット & 診断プロット)
    # ---------------------------------------------------------
    df_ensemble_pred = df_preds_all.groupby(level=0).mean(numeric_only=True).drop(columns=['Seed'], errors='ignore')
    for model in models:
        df_preds_all[f'AbsError_{model}'] = df_preds_all[f'Error_{model}'].abs()
        df_ensemble_pred[f'Mean_L1_Loss_{model}'] = df_preds_all.groupby(level=0)[f'AbsError_{model}'].mean()
    df_ensemble_pred.to_csv(os.path.join(SUMMARY_DIR, 'ensemble_mean_predictions.csv'), encoding='utf-8-sig')

    print("\n[Generating Plots and Aggregating Diagnostic Metrics]")
    group_folders = [TARGET_DIR] + ([os.path.abspath(d) for d in args.group_dirs] if args.group_dirs else [])
    group_vals_list = [load_group_pred_values(f) for f in group_folders]
    all_group_vals = np.concatenate([v for v in group_vals_list if len(v) > 0])

    min_pred = float(np.nanmin(all_group_vals))
    max_pred = float(np.nanmax(all_group_vals))
    margin_pred = (max_pred - min_pred) * 0.05 if max_pred != min_pred else 10.0
    y_lim_preds = (min_pred - margin_pred, max_pred + margin_pred)
    print(f"-> Unified Y-axis range for prediction plots: {y_lim_preds[0]:.2f} - {y_lim_preds[1]:.2f}")

    # Figure 3 ペア比較プロット (Model A vs C, Model B vs D) の出力
    generate_publication_figure3_plots(df_preds_all, df_ensemble_pred, SUMMARY_DIR, y_lim_preds)

    all_d_bars, all_cum_d_bars, all_cum_d_tks = [], [], []
    for base_m, adv_m in comparison_pairs:
        d_b = df_ensemble_pred[f'Mean_L1_Loss_{base_m}'] - df_ensemble_pred[f'Mean_L1_Loss_{adv_m}']
        all_d_bars.append(d_b)
        all_cum_d_bars.append(d_b.cumsum())
        preds_adv = df_preds_all.reset_index().pivot(index='Date', columns='Seed', values=f'Pred_{adv_m}')
        for seed in preds_adv.columns:
            df_s = df_preds_all[df_preds_all['Seed'] == seed].sort_index()
            d_tk = np.abs(df_s[f'Error_{base_m}']) - np.abs(df_s[f'Error_{adv_m}'])
            all_cum_d_tks.append(d_tk.cumsum())

    global_d_bar = pd.concat(all_d_bars)
    min_diff, max_diff = global_d_bar.min(), global_d_bar.max()
    margin_diff = (max_diff - min_diff) * 0.05 if max_diff != min_diff else 10
    y_lim_diff = (min_diff - margin_diff, max_diff + margin_diff)

    global_cum = pd.concat(all_cum_d_bars + all_cum_d_tks)
    min_cum, max_cum = global_cum.min(), global_cum.max()
    margin_cum = (max_cum - min_cum) * 0.05 if max_cum != min_cum else 50
    y_lim_cum = (min_cum - margin_cum, max_cum + margin_cum)

    for model in models:
        generate_prediction_bands(df_preds_all, df_ensemble_pred, model, SUMMARY_DIR, y_lim_preds)

    all_diagnostics = []
    for base_m, adv_m in comparison_pairs:
        print(f"  Generating plots & metrics for {adv_m} vs {base_m} ...")
        metrics = generate_diagnostic_plots(df_preds_all, df_ensemble_pred, base_m, adv_m, SUMMARY_DIR, y_lim_diff, y_lim_cum)
        all_diagnostics.append(metrics)

    df_diagnostics = pd.DataFrame(all_diagnostics)
    df_diagnostics.to_csv(os.path.join(SUMMARY_DIR, 'diagnostic_metrics_all_pairs.csv'), index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------
    # 4. モデル安定性評価 (model_stability_metrics)
    # ---------------------------------------------------------
    stability_results = []
    for model in models:
        pred_matrix = df_preds_all.reset_index().pivot(index='Date', columns='Seed', values=f'Pred_{model}')
        corr_matrix = pred_matrix.corr()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        mean_corr = upper_tri.mean().mean()

        df_model_logs = df_logs_all[df_logs_all['Model'] == model]
        jaccard_scores = []
        for date in dates:
            df_date = df_model_logs[df_model_logs['Date'] == date]
            sets = [set(f.split(', ')) if pd.notna(f) and f else set() for f in df_date['Selected_Features']]
            if len(sets) > 1:
                j_vals = [jaccard_index(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
                jaccard_scores.append(np.mean(j_vals))
        mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else np.nan

        stability_results.append({
            'Model': model,
            'Mean_Pred_Correlation': round(mean_corr, 4),
            'Mean_Jaccard_Index': round(mean_jaccard, 4)
        })

    df_stability = pd.DataFrame(stability_results)
    print("\n[Model Stability Metrics across Seeds]")
    print(df_stability.to_string(index=False))
    df_stability.to_csv(os.path.join(SUMMARY_DIR, 'model_stability_metrics.csv'), index=False, encoding='utf-8-sig')

    # ---------------------------------------------------------
    # 5. SHAP用代表シード選定 (representative_seeds)
    # ---------------------------------------------------------
    rep_seeds = []
    for model in models:
        ensemble_pred = df_ensemble_pred[f'Pred_{model}']
        best_seed, min_diff_mae = None, float('inf')
        for seed in seeds:
            seed_pred = df_preds_all[df_preds_all['Seed'] == seed][f'Pred_{model}']
            diff_mae = (ensemble_pred - seed_pred).abs().mean()
            if diff_mae < min_diff_mae:
                min_diff_mae = diff_mae
                best_seed = seed
        rep_seeds.append({'Model': model, 'Representative_Seed': best_seed, 'Diff_from_Centroid_MAE': round(min_diff_mae, 4)})

    df_rep_seeds = pd.DataFrame(rep_seeds)
    print("\n[Representative Seeds for SHAP Analysis]")
    print(df_rep_seeds.to_string(index=False))
    df_rep_seeds.to_csv(os.path.join(SUMMARY_DIR, 'representative_seeds.csv'), index=False, encoding='utf-8-sig')

    print(f"\nAll outputs successfully saved in: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
