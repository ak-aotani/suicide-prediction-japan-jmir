# ===================================================================
# LightGBM 10-Seed Global & Local SHAP Analysis Pipeline
# File: code_shap_10seed.py
# ===================================================================
import pandas as pd
import numpy as np
import os
import sys
import argparse
import glob
import pickle


def export_local_shap_breakdown(shap_series, df_preds_all, rep_seed, model, target_date, perspective, out_dir, cutoff_pct=90):
    """
    Extracts Local SHAP values for a specific target month using a cumulative
    absolute SHAP threshold (default: 90%), attaches prediction metadata
    (actual value, predicted value, prediction error, and base value),
    separates positive and negative contributions, and exports the results to CSV.
    """
    # Retrieve prediction metadata for the representative seed at the target date
    seed_data = df_preds_all[df_preds_all['Seed'] == rep_seed].sort_index()
    actual_val = float(seed_data.loc[target_date, 'Actual'])
    pred_val = float(seed_data.loc[target_date, f'Pred_{model}'])
    error_val = float(seed_data.loc[target_date, f'Error_{model}'])

    # Reconstruct Base Value (Predicted Value = Base Value + Sum of SHAP values)
    sum_shap = float(shap_series.sum())
    base_value = pred_val - sum_shap

    # Compute cumulative percentage based on absolute SHAP values
    abs_shap = shap_series.abs().sort_values(ascending=False)
    total_abs_shap = abs_shap.sum()

    if total_abs_shap == 0:
        # Fallback if all SHAP values are zero
        top_shap = shap_series.copy()
        bottom_shap = pd.Series(dtype=float)
    else:
        cumulative_pct = (abs_shap.cumsum() / total_abs_shap) * 100.0
        cutoff_idx = cumulative_pct[cumulative_pct < cutoff_pct].index.tolist()

        if len(cutoff_idx) < len(abs_shap):
            next_var = abs_shap.index[len(cutoff_idx)]
            cutoff_idx.append(next_var)

        top_shap = shap_series.loc[cutoff_idx].copy()
        bottom_shap = shap_series.drop(index=cutoff_idx)

    # Separate into positive and negative contributions and sort by magnitude
    pos_top = top_shap[top_shap > 0].sort_values(ascending=False)
    neg_top = top_shap[top_shap < 0].sort_values(ascending=True)  # Sorted from most negative

    # Aggregate remaining features below the 90% cutoff by sign
    if not bottom_shap.empty:
        others_pos = float(bottom_shap[bottom_shap > 0].sum())
        others_neg = float(bottom_shap[bottom_shap < 0].sum())

        if others_pos > 0:
            pos_top['Others_Positive'] = others_pos
        if others_neg < 0:
            neg_top['Others_Negative'] = others_neg

    # Export breakdown to CSV
    date_str = target_date.strftime('%Y-%m-%d')
    out_filename = f"shap90_{perspective}_{date_str}_{model}.csv"
    out_path = os.path.join(out_dir, out_filename)

    with open(out_path, 'w', encoding='utf-8-sig') as f:
        # Metadata block
        f.write("--- METADATA ---\n")
        f.write(f"Model,{model}\n")
        f.write(f"Target Date,{date_str}\n")
        f.write(f"Representative Seed,{rep_seed}\n")
        f.write(f"Actual Value,{actual_val:.4f}\n")
        f.write(f"Predicted Value,{pred_val:.4f}\n")
        f.write(f"Error (Actual - Pred),{error_val:.4f}\n")
        f.write(f"Base Value,{base_value:.4f}\n")
        f.write("\n")

        # SHAP contribution block
        f.write("Group,Feature,SHAP_Value\n")
        for k, v in pos_top.items():
            f.write(f"Positive,{k},{v:.4f}\n")
        for k, v in neg_top.items():
            f.write(f"Negative,{k},{v:.4f}\n")

    print(f" -> Saved: {out_filename}")


def main():
    parser = argparse.ArgumentParser(description="Run 10-seed Global and Local SHAP analysis with a 90% cumulative importance cutoff.")
    parser.add_argument("target_dir", type=str, help="Path to the master output directory containing seed_* folders.")
    parser.add_argument("--target_month", type=str, default=None, help="Optional: Specific target month for Local SHAP breakdown (YYYY-MM-DD).")
    args = parser.parse_args()

    TARGET_DIR = os.path.abspath(args.target_dir)
    TARGET_MONTH = args.target_month

    if not os.path.isdir(TARGET_DIR):
        print(f"Error: Directory {TARGET_DIR} not found.")
        sys.exit(1)

    SHAP_DIR = os.path.join(TARGET_DIR, 'shap_analysis')
    os.makedirs(SHAP_DIR, exist_ok=True)

    # 1. Load prediction results across all seeds and compute the 10-seed ensemble mean
    seed_dirs = sorted([d for d in glob.glob(os.path.join(TARGET_DIR, 'seed_*')) if os.path.isdir(d)])
    all_preds = []
    for seed_dir in seed_dirs:
        try:
            seed_num = int(os.path.basename(seed_dir).split('_')[1])
        except (IndexError, ValueError):
            continue
        pred_file = os.path.join(seed_dir, 'all_models_forecast_comparison.csv')
        if os.path.exists(pred_file):
            df_pred = pd.read_csv(pred_file, parse_dates=['Date'], index_col='Date')
            df_pred['Seed'] = seed_num
            all_preds.append(df_pred)

    if not all_preds:
        print(f"Error: No valid seed prediction files found in {TARGET_DIR}.")
        sys.exit(1)

    df_preds_all = pd.concat(all_preds)
    models = [m.replace('Pred_', '') for m in df_preds_all.columns if m.startswith('Pred_')]
    seeds = sorted(df_preds_all['Seed'].unique())
    df_ensemble = df_preds_all.groupby(level=0).mean(numeric_only=True)

    # 2. Select the representative seed for each model (seed minimizing MAE relative to the 10-seed ensemble mean)
    rep_seeds = {}
    for model in models:
        min_diff = float('inf')
        best_seed = None
        for seed in seeds:
            df_seed = df_preds_all[df_preds_all['Seed'] == seed].sort_index()
            diff = (df_seed[f'Pred_{model}'] - df_ensemble[f'Pred_{model}']).abs().mean()
            if diff < min_diff:
                min_diff = diff
                best_seed = seed
        rep_seeds[model] = best_seed

    print("\n[Representative Seeds for Local SHAP Extraction]")
    for m, s in rep_seeds.items():
        print(f"{m}: Seed {s}")

    # 3. Compute Global SHAP values (averaged across all 36 validation months and 10 seeds) and apply the 90% cutoff
    CUTOFF_PCT = 90
    for model in models:
        seed_mean_abs_shap_list = []
        for seed in seeds:
            pkl_path = os.path.join(TARGET_DIR, f'seed_{seed}', f'shap_artifacts_{model}.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                mean_abs_shap = np.abs(data['all_shap_values']).mean(axis=0)
                seed_mean_abs_shap_list.append(pd.Series(mean_abs_shap, index=data['feature_names']))

        if seed_mean_abs_shap_list:
            df_global = pd.DataFrame(seed_mean_abs_shap_list).mean(axis=0).sort_values(ascending=False)
            df_global.name = 'Mean_Abs_SHAP'

            # Export full Global SHAP ranking
            df_global.to_csv(os.path.join(SHAP_DIR, f'global_shap_importance_{model}_full.csv'), index_label='Feature', encoding='utf-8-sig')

            # Extract top features accounting for 90% of cumulative absolute SHAP importance and aggregate the rest as "Others"
            total_shap = df_global.sum()
            cumulative_pct = (df_global.cumsum() / total_shap) * 100.0

            cutoff_idx = cumulative_pct[cumulative_pct < CUTOFF_PCT].index.tolist()
            if len(cutoff_idx) < len(df_global):
                next_var = df_global.index[len(cutoff_idx)]
                cutoff_idx.append(next_var)

            df_top = df_global.loc[cutoff_idx].copy()
            others_sum = df_global.drop(index=cutoff_idx).sum()

            if others_sum > 0:
                df_top.loc['Others'] = others_sum

            df_top.to_csv(os.path.join(SHAP_DIR, f'global_shap_importance_{model}_top{CUTOFF_PCT}pct.csv'), index_label='Feature', encoding='utf-8-sig')

    # 4. Load Local SHAP time-series matrices for the representative seeds
    local_shap_ts_dict = {}
    for model in models:
        rep_seed = rep_seeds[model]
        pkl_path = os.path.join(TARGET_DIR, f'seed_{rep_seed}', f'shap_artifacts_{model}.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            df_shap_ts = pd.DataFrame(
                data['all_shap_values'],
                columns=data['feature_names'],
                index=data['all_features_df'].index
            )
            local_shap_ts_dict[model] = df_shap_ts

    # 5. Identify notable validation months and compile target months for Local SHAP breakdown
    df_rep_preds = pd.DataFrame(index=df_ensemble.index)
    df_rep_preds['Actual'] = df_preds_all.groupby(level=0)['Actual'].first()
    for model in models:
        rep_seed = rep_seeds[model]
        seed_data = df_preds_all[df_preds_all['Seed'] == rep_seed].sort_index()
        df_rep_preds[f'Pred_{model}'] = seed_data[f'Pred_{model}']
        df_rep_preds[f'Error_{model}'] = seed_data[f'Error_{model}'].abs()

    insights = []
    top1_targets = []  # List of tuples: (perspective_label, target_date, model_list)

    # Perspective A: Best prediction months (smallest absolute error) for each model
    for model in models:
        top3 = df_rep_preds[f'Error_{model}'].nsmallest(3)
        for rank, (date, err) in enumerate(top3.items(), 1):
            insights.append({'Perspective': f'{model} Best', 'Rank': rank, 'Date': date, 'Value (Abs Error)': err})
            if rank == 1:
                top1_targets.append((f'{model}_Best', date, [model]))

    # Perspective B: Worst prediction months (largest absolute error) for each model
    for model in models:
        worst3 = df_rep_preds[f'Error_{model}'].nlargest(3)
        for rank, (date, err) in enumerate(worst3.items(), 1):
            insights.append({'Perspective': f'{model} Worst', 'Rank': rank, 'Date': date, 'Value (Abs Error)': err})
            if rank == 1:
                top1_targets.append((f'{model}_Worst', date, [model]))

    # Perspective C: Months where Model C outperformed Model D by the largest margin
    if 'Model_C' in models and 'Model_D' in models:
        c_wins_margin = df_rep_preds['Error_Model_D'] - df_rep_preds['Error_Model_C']
        c_wins_top3 = c_wins_margin.nlargest(3)
        for rank, (date, margin) in enumerate(c_wins_top3.items(), 1):
            insights.append({'Perspective': 'Model C beats Model D', 'Rank': rank, 'Date': date, 'Value (Error Diff)': margin})
            if rank == 1:
                top1_targets.append(('C_vs_D', date, ['Model_C', 'Model_D']))

    # Perspective D: Months where Model C outperformed Model A by the largest margin (eg, May 2024)
    if 'Model_C' in models and 'Model_A' in models:
        c_beats_a_margin = df_rep_preds['Error_Model_A'] - df_rep_preds['Error_Model_C']
        c_beats_a_top3 = c_beats_a_margin.nlargest(3)
        for rank, (date, margin) in enumerate(c_beats_a_top3.items(), 1):
            insights.append({'Perspective': 'Model C beats Model A', 'Rank': rank, 'Date': date, 'Value (Error Diff)': margin})
            if rank == 1:
                top1_targets.append(('C_vs_A', date, ['Model_C', 'Model_A']))

    # Perspective E: User-specified custom target month via command-line argument
    if TARGET_MONTH:
        try:
            custom_date = pd.to_datetime(TARGET_MONTH)
            if custom_date in df_rep_preds.index:
                top1_targets.append(('Custom_Month', custom_date, models))
                print(f"\n[Custom Month Detected: {TARGET_MONTH}] Added to extraction targets.")
        except Exception as e:
            print(f"\n[Warning] Could not parse --target_month '{TARGET_MONTH}'. Ignoring. Error: {e}")

    df_insights = pd.DataFrame(insights)
    df_insights.to_csv(os.path.join(SHAP_DIR, 'interesting_months_list.csv'), index=False, encoding='utf-8-sig')

    # 6. Export Local SHAP breakdown CSVs for the top-1 target months
    print("\n[Extracting Top 1 Local SHAP Profiles (90% Cutoff with Positive/Negative Split)]")
    for perspective, date, target_models in top1_targets:
        for model in target_models:
            if model in local_shap_ts_dict and date in local_shap_ts_dict[model].index:
                shap_row = local_shap_ts_dict[model].loc[date]
                rep_seed = rep_seeds[model]

                export_local_shap_breakdown(
                    shap_series=shap_row,
                    df_preds_all=df_preds_all,
                    rep_seed=rep_seed,
                    model=model,
                    target_date=date,
                    perspective=perspective,
                    out_dir=SHAP_DIR,
                    cutoff_pct=CUTOFF_PCT
                )

    print(f"\nSHAP Analysis complete. Results saved in: {SHAP_DIR}")


if __name__ == "__main__":
    main()
