# ===================================================================
# Hybrid Dynamic Walk-Forward Validation: 10-Seed Computation Script
# ===================================================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import pickle
import os
import sys
import argparse
import re
import random
import hashlib
import importlib.metadata
import shutil
from datetime import datetime
import warnings
import shap
import logging

# --- ライブラリ設定 ---
warnings.filterwarnings(
    'ignore',
    category=UserWarning,
    message=re.escape("X does not have valid feature names, but LGBMRegressor was fitted with feature names")
)
optuna.logging.set_verbosity(optuna.logging.INFO)
optuna.logging.disable_default_handler()

# ===================================================================
# Reproducibility & Utility Functions
# ===================================================================
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def get_file_hash(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def save_environment_info(output_dir):
    env_path = os.path.join(output_dir, 'requirements_env.txt')
    with open(env_path, 'w', encoding='utf-8') as f:
        dists = sorted(importlib.metadata.distributions(), key=lambda x: x.metadata['Name'])
        for dist in dists:
            f.write(f"{dist.metadata['Name']}=={dist.version}\n")

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ===================================================================
# Settings & Model Configurations
# ===================================================================
# 10個の固定シード値（再現性・安定性検証用）
SEEDS = [42, 123, 456, 789, 1024, 2048, 3072, 4096, 8192, 9999]

TARGET_COLUMN = 'suicide'
PREDICTION_HORIZON = 2
NULL_IMP_HALFLIFE_MONTHS = 36
VALIDATION_START = '2022-01-01'
VALIDATION_END = '2024-12-31'
OPTUNA_VAL_WINDOW_YEARS = 3
TUNING_TRIALS = 30  # 30回に制限（探索空間の純化による最適化）
CORR_THRESHOLD = 0.85

STATIC_INT_LAGS = [2, 3, 4, 6, 12, 24]
STATIC_EXT_LAGS = [2, 3, 4, 6, 12, 24]

# 構造的制約の完全固定化 (深さ3統一・葉数制限)
BASE_LGBM_PARAMS = {
    'objective': 'regression_l1',
    'metric': 'mae',
    'n_estimators': 400,
    'learning_rate': 0.05,
    'max_depth': 3,
    'num_leaves': 7,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_child_samples': 15,
    'linear_tree': False,
    'verbose': -1,
    'n_jobs': -1,
    'deterministic': True,
    'force_row_wise': True
}

MODELS_CONFIG = {
    'Model_A': {'desc': 'Simple Internal (Lags only)',       'use_external': False, 'use_rolling': False},
    'Model_B': {'desc': 'Advanced Internal (Lags+Rolling)',  'use_external': False, 'use_rolling': True},
    'Model_C': {'desc': 'Simple + External (Ext Lags)',      'use_external': True,  'use_rolling': False},
    'Model_D': {'desc': 'Advanced + External (Full Model)',  'use_external': True,  'use_rolling': True}
}

# ===================================================================
# Function Definitions
# ===================================================================
def generate_discrete_features(df_input, target_name, int_lags, ext_lags, use_external, use_rolling):
    all_features_list = []
    pool = []
    date_idx = df_input.index
    date_features = pd.DataFrame({
        'year': date_idx.year, 'month': date_idx.month,
        'month_sin': np.sin(2 * np.pi * date_idx.month / 12),
        'month_cos': np.cos(2 * np.pi * date_idx.month / 12)
    }, index=date_idx)
    all_features_list.append(date_features)
    pool.extend(date_features.columns.tolist())

    internal_concat = []
    for lag in int_lags:
        internal_concat.append(df_input[target_name].shift(lag).rename(f'target_level_lag{lag:02d}'))
        internal_concat.append(df_input[target_name].shift(lag).diff(1).rename(f'target_diff1_lag{lag:02d}'))

    if use_rolling:
        shifted_diff = df_input[target_name].shift(PREDICTION_HORIZON).diff(1)
        for window in [3, 6, 12]:
            internal_concat.append(shifted_diff.rolling(window=window).mean().rename(f'target_rolling_mean_diff1_w{window}_lag{PREDICTION_HORIZON:02d}'))
            internal_concat.append(shifted_diff.rolling(window=window).std().rename(f'target_rolling_std_diff1_w{window}_lag{PREDICTION_HORIZON:02d}'))

    df_internal = pd.concat(internal_concat, axis=1)
    pool.extend(df_internal.columns.tolist())
    all_features_list.append(df_internal)

    if use_external:
        external_concat = []
        external_vars = [col for col in df_input.columns if col != target_name]
        for col in external_vars:
            match = re.match(r'lagcap(\d+)_', col)
            lag_cap = int(match.group(1)) if match else PREDICTION_HORIZON
            clean_col_name = col[len(match.group(0)):] if match else col

            target_lags = sorted(list(set(ext_lags + [lag_cap])))
            for lag in target_lags:
                if lag >= lag_cap:
                    external_concat.append(df_input[col].shift(lag).diff(1).rename(f'{clean_col_name}_diff1_lag{lag:02d}'))

        if external_concat:
            df_external = pd.concat(external_concat, axis=1)
            pool.extend(df_external.columns.tolist())
            all_features_list.append(df_external)

    df_final = pd.concat([df_input[[target_name]], pd.concat(all_features_list, axis=1)], axis=1)
    return df_final, sorted(list(set(pool)))

def filter_highly_correlated_features(X, y, threshold):
    # 標準偏差が極小（実質定数）の列および欠損を含む列を事前に除外
    valid_cols = X.columns[(X.std(ddof=1) > 1e-8) & (X.notna().all())]
    if len(valid_cols) == 0:
        return X[[]]

    X_active = X[valid_cols]
    target_corr = X_active.corrwith(y).abs().fillna(0.0)
    corr_matrix = X_active.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    for col_right in upper.columns:
        for col_left in upper.index:
            if upper.loc[col_left, col_right] > threshold:
                if col_left in to_drop or col_right in to_drop: continue
                if target_corr.get(col_left, 0) < target_corr.get(col_right, 0): to_drop.add(col_left)
                else: to_drop.add(col_right)
    return X_active.drop(columns=list(to_drop))

def apply_null_importance(X_train, y_train, base_params, num_runs=50, percentile=80, halflife=NULL_IMP_HALFLIFE_MONTHS, current_seed=42):
    if X_train.shape[1] == 0:
        print(f"  [Warning] No candidate features passed to Null Importance (Seed: {current_seed}). Proceeding with fixed calendar features only.")
        return []

    n_samples = len(y_train)
    distances = np.arange(n_samples - 1, -1, -1)
    sample_weights = (0.5) ** (distances / halflife)
    model_actual = lgb.LGBMRegressor(**base_params)
    model_actual.fit(X_train, y_train, sample_weight=sample_weights)
    actual_imp = pd.Series(model_actual.feature_importances_, index=X_train.columns)

    null_imp_list = []
    y_values = y_train.values
    for i in range(num_runs):
        # グローバルな np.random を上書きせず、独立した乱数生成器インスタンスを使用
        rng = np.random.RandomState(current_seed * 10000 + i)
        shuffled_y = rng.permutation(y_values)

        model_null = lgb.LGBMRegressor(**base_params)
        model_null.fit(X_train, shuffled_y, sample_weight=sample_weights)
        null_imp_list.append(model_null.feature_importances_)

    null_imp_df = pd.DataFrame(null_imp_list, columns=X_train.columns)
    survived = [col for col in X_train.columns if actual_imp[col] > np.percentile(null_imp_df[col], percentile)]
    if len(survived) == 0:
        print(f"  [Warning] Null Importance rejected all candidate features (Seed: {current_seed}). Proceeding with fixed calendar features only.")
    return survived

def optimize_hyperparameters(df_master, target_col, base_pool_features, current_val_date, h, current_seed):
    opt_val_end = current_val_date - pd.DateOffset(months=h)
    opt_val_start = opt_val_end - pd.DateOffset(years=OPTUNA_VAL_WINDOW_YEARS) + pd.DateOffset(months=1)
    train_data_init = df_master.loc[:(opt_val_start - pd.DateOffset(months=h))]
    y_init = train_data_init[target_col].dropna()
    fixed_features = ['year', 'month', 'month_sin', 'month_cos']
    X_init = train_data_init.loc[y_init.index, [f for f in base_pool_features if f not in fixed_features]].dropna()
    y_init = y_init.loc[X_init.index]

    X_init_filtered = filter_highly_correlated_features(X_init, y_init, CORR_THRESHOLD)

    null_imp_params = BASE_LGBM_PARAMS.copy()
    null_imp_params.update({'reg_alpha': 0.01, 'seed': current_seed})
    survived_candidates = apply_null_importance(X_init_filtered, y_init, null_imp_params, current_seed=current_seed)
    optuna_pool = sorted(list(set(fixed_features + survived_candidates)))

    def objective(trial):
        model_params_tuning = BASE_LGBM_PARAMS.copy()
        model_params_tuning.update({
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1e2, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1e2, log=True),
            'seed': current_seed
        })
        _, avg_mae = run_walk_forward_dynamic(
            df_master=df_master, target_col=target_col, base_pool_features=optuna_pool,
            val_start=opt_val_start, val_end=opt_val_end, h=h,
            model_params=model_params_tuning, output_dir=None, model_name="", is_tuning=True, apply_dynamic_null_imp=False, current_seed=current_seed
        )
        return avg_mae

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=current_seed))
    study.optimize(objective, n_trials=TUNING_TRIALS)
    return study.best_params

def run_walk_forward_dynamic(df_master, target_col, base_pool_features, val_start, val_end, h, model_params, output_dir=None, model_name="", is_tuning=False, apply_dynamic_null_imp=False, current_seed=42):
    validation_range = pd.date_range(start=val_start, end=val_end, freq='MS')
    predictions, all_results, monthly_artifacts_list = {}, [], []
    all_shap_values_df_list, all_features_df_list = [], []
    feature_counts = []

    fixed_features = ['year', 'month', 'month_sin', 'month_cos']
    current_best_params = model_params.copy()
    current_best_params['seed'] = current_seed

    for i, current_val_date in enumerate(validation_range):
        train_end_date = current_val_date - pd.DateOffset(months=h)

        if not is_tuning and (current_val_date.month == 1 or i == 0):
            updated_params = optimize_hyperparameters(df_master, target_col, base_pool_features, current_val_date, h, current_seed)
            current_best_params.update({
                'reg_alpha': updated_params['reg_alpha'],
                'reg_lambda': updated_params['reg_lambda']
            })

        train_data = df_master.loc[:train_end_date]
        y_train = train_data[target_col].dropna()
        X_train_full = train_data.loc[y_train.index, base_pool_features].dropna()
        y_train = y_train.loc[X_train_full.index]

        active_pool = base_pool_features
        if apply_dynamic_null_imp and not is_tuning:
            X_train_candidates = X_train_full[[f for f in base_pool_features if f not in fixed_features]]
            X_train_filtered = filter_highly_correlated_features(X_train_candidates, y_train, CORR_THRESHOLD)

            null_imp_params = BASE_LGBM_PARAMS.copy()
            null_imp_params.update({'reg_alpha': 0.01, 'seed': current_seed})
            survived_candidates = apply_null_importance(X_train_filtered, y_train, null_imp_params, current_seed=current_seed)
            active_pool = sorted(list(set(fixed_features + survived_candidates)))

        X_train_final = X_train_full[active_pool]
        model = lgb.LGBMRegressor(**current_best_params)
        model.fit(X_train_final, y_train)

        active_features = X_train_final.columns[model.feature_importances_ > 0].tolist()
        feature_counts.append(len(active_features))

        X_to_predict = df_master.loc[[current_val_date], active_pool]
        y_pred = model.predict(X_to_predict)[0]
        predictions[current_val_date] = y_pred
        actual_val = df_master.loc[current_val_date, target_col]

        if is_tuning: continue

        all_results.append({
            "Date": current_val_date, "actual": actual_val, "prediction": y_pred,
            "error": actual_val - y_pred, "Num_Features": len(active_features),
            "Selected_Features": ', '.join(sorted(active_features)),
            "Params_Depth": current_best_params.get('max_depth', np.nan),
            "Params_Alpha": current_best_params.get('reg_alpha', np.nan),
            "Params_Lambda": current_best_params.get('reg_lambda', np.nan)
        })

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_to_predict, check_additivity=False)
            shap_series = pd.Series(shap_values[0] if shap_values.ndim == 2 else shap_values, index=X_to_predict.columns)
            shap_df = pd.DataFrame(0.0, index=[current_val_date], columns=base_pool_features)
            for f, v in shap_series.items():
                if f in shap_df.columns: shap_df.at[current_val_date, f] = v
            all_shap_values_df_list.append(shap_df)

            # 全候補特徴量(base_pool_features)の実測値を保存して列構造を一致させる
            all_features_df_list.append(df_master.loc[[current_val_date], base_pool_features])
        except Exception as e:
            print(f"  [Warning] SHAP calculation failed at {current_val_date}: {e}")

        monthly_artifacts_list.append({'date': current_val_date, 'model': model, 'X_train': X_train_final})

    pred_series = pd.Series(predictions)
    actual_series = df_master.loc[pred_series.index, target_col]

    if is_tuning:
        return (np.mean(feature_counts) if feature_counts else 0), (actual_series - pred_series).abs().mean()
    else:
        if output_dir and model_name:
            try:
                with open(os.path.join(output_dir, f'monthly_models_{model_name}.pkl'), 'wb') as f:
                    pickle.dump(monthly_artifacts_list, f)
                if all_shap_values_df_list:
                    with open(os.path.join(output_dir, f'shap_artifacts_{model_name}.pkl'), 'wb') as f:
                        pickle.dump({'all_shap_values': pd.concat(all_shap_values_df_list).values, 'feature_names': pd.concat(all_shap_values_df_list).columns.tolist(), 'all_features_df': pd.concat(all_features_df_list)}, f)
            except Exception as e:
                print(f"Artifact save error: {e}")

        return actual_series, pred_series, (actual_series - pred_series), pd.DataFrame(all_results)

# ===================================================================
# Main Execution Block (10-Seed Loop)
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 10-Seed computation for Models A-D.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file.")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FILE_PATH = os.path.join(BASE_DIR, args.input_file)

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    dataset_name = os.path.splitext(os.path.basename(args.input_file))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    MASTER_OUTPUT_DIR = os.path.join(BASE_DIR, f"{script_name}_{dataset_name}_{timestamp}")
    os.makedirs(MASTER_OUTPUT_DIR, exist_ok=True)

    try:
        script_path = os.path.abspath(__file__)
        shutil.copy(script_path, os.path.join(MASTER_OUTPUT_DIR, os.path.basename(script_path)))
    except Exception:
        pass

    save_environment_info(MASTER_OUTPUT_DIR)

    sys.stdout = Logger(os.path.join(MASTER_OUTPUT_DIR, f'execution_log_ensemble_{timestamp}.txt'))
    sys.stderr = sys.stdout
    optuna_logger = logging.getLogger("optuna")
    optuna_logger.setLevel(logging.WARNING)
    optuna_handler = logging.StreamHandler(sys.stdout)
    optuna_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    optuna_logger.addHandler(optuna_handler)

    try:
        if not os.path.exists(FILE_PATH): raise FileNotFoundError(f"Input file not found at {FILE_PATH}")

        print(f"========== Starting 10-Seed Computation Pipeline ==========")
        file_hash = get_file_hash(FILE_PATH)
        print(f"Target file: {args.input_file}")
        print(f"File SHA-256 Checksum: {file_hash}\n")

        df_master_raw = pd.read_csv(FILE_PATH, parse_dates=[0], index_col=0)

        for current_seed in SEEDS:
            print(f"\n" + "="*60)
            print(f"Starting execution for SEED: {current_seed}")
            print(f"="*60)

            # 各シード固有のサブディレクトリを作成
            seed_dir = os.path.join(MASTER_OUTPUT_DIR, f'seed_{current_seed}')
            os.makedirs(seed_dir, exist_ok=True)

            # 実行単位でのシード固定
            seed_everything(current_seed)

            all_predictions_df = pd.DataFrame()
            all_detailed_logs_list = []

            for model_name, config in MODELS_CONFIG.items():
                print(f"\nExecuting {model_name}: {config['desc']} (Seed: {current_seed})")
                print(f"-----------------------------------------------------------")

                df_features, full_pool = generate_discrete_features(
                    df_master_raw, TARGET_COLUMN, STATIC_INT_LAGS, STATIC_EXT_LAGS,
                    use_external=config['use_external'], use_rolling=config['use_rolling']
                )

                initial_params = BASE_LGBM_PARAMS.copy()
                initial_params.update({'reg_alpha': 0.1, 'reg_lambda': 0.1, 'seed': current_seed})

                actual, pred, errors, detailed_log = run_walk_forward_dynamic(
                    df_master=df_features, target_col=TARGET_COLUMN, base_pool_features=full_pool,
                    val_start=VALIDATION_START, val_end=VALIDATION_END, h=PREDICTION_HORIZON,
                    model_params=initial_params, output_dir=seed_dir, model_name=model_name, is_tuning=False, apply_dynamic_null_imp=True, current_seed=current_seed
                )

                detailed_log.insert(0, 'Model', model_name)
                all_detailed_logs_list.append(detailed_log)

                mae = errors.abs().mean()
                print(f"  [Finished] {model_name} - MAE: {mae:.4f}")

                if all_predictions_df.empty:
                    all_predictions_df['Date'] = pred.index
                    all_predictions_df['Actual'] = actual.values
                    all_predictions_df.set_index('Date', inplace=True)

                all_predictions_df[f'Pred_{model_name}'] = pred.values
                all_predictions_df[f'Error_{model_name}'] = errors.values
                all_predictions_df[f'NumFeatures_{model_name}'] = detailed_log['Num_Features'].values

            # シードごとの結果をシード専用ディレクトリに保存
            all_predictions_df.to_csv(os.path.join(seed_dir, 'all_models_forecast_comparison.csv'))
            combined_detailed_log_df = pd.concat(all_detailed_logs_list, ignore_index=True)
            combined_detailed_log_df.to_csv(os.path.join(seed_dir, 'walk_forward_detailed_log_all.csv'), index=False)
            print(f"\nSaved results for SEED {current_seed} in {seed_dir}")

        print(f"\n========== All 10 Seeds Computation Complete ==========")
        print(f"Master Output Directory: {MASTER_OUTPUT_DIR}")

    except Exception as e:
        import traceback
        print(f"\n[Error] 予期せぬエラーが発生しました: {e}")
        traceback.print_exc()
