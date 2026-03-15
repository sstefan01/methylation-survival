import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import os
import sys
from scipy.interpolate import interp1d
from typing import List, Optional, Tuple
import warnings
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Third-party imports ---
try: from lifelines.utils import concordance_index
except ImportError: print("Error: 'lifelines' package not found."); sys.exit(1)
try: from sksurv.metrics import cumulative_dynamic_auc; from sksurv.util import Surv
except ImportError: print("Error: 'scikit-survival' package not found."); sys.exit(1)
try: from sklearn.metrics import brier_score_loss
except ImportError: print("Error: 'scikit-learn' package not found."); sys.exit(1)

# --- Project specific imports ---
try: from utils import load_survival_data
except ImportError as e: print(f"Error importing project modules: {e}"); sys.exit(1)
except Exception as e: print(f"An unexpected error during project import: {e}"); sys.exit(1)

# --- Brier Score Helper Functions ---
def compute_metric_at_times(metric, time_true, prob_pred, event_observed, score_times):
    scores = {}; event_observed = event_observed.astype(bool)
    for i, time in enumerate(score_times):
        if i >= prob_pred.shape[1]: scores[time] = np.nan; continue
        pred_at_time = prob_pred[:, i]; valid_mask = (event_observed & (time_true <= time)) | (~event_observed & (time_true >= time))
        if not valid_mask.any(): scores[time] = np.nan; continue
        true_outcome_at_time = (time_true > time).astype(int)
        try: score = metric(true_outcome_at_time[valid_mask], pred_at_time[valid_mask])
        except ValueError as e: score = np.nan; print(f"Warn: Brier@T={time:.2f}: {e}")
        scores[time] = score
    return scores

def brier_score_at_times(time_true, prob_pred, event_observed, score_times):
    return compute_metric_at_times(brier_score_loss, time_true, prob_pred, event_observed, score_times)

# --- Config Loader ---
def load_config(config_path):
    if not os.path.exists(config_path): raise FileNotFoundError(f"Config not found: {config_path}")
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f: config = yaml.safe_load(f)
        return config
    except Exception as e: print(f"Error loading/parsing config {config_path}: {e}"); raise

# --- Main Evaluation Function ---
def main(args):
    # --- 1. Load Configuration ---
    config = load_config(args.config)
    pred_path = args.pred_path

    # --- 2. Load Predictions CSV ---
    if not os.path.exists(pred_path): raise FileNotFoundError(f"Prediction CSV not found: {pred_path}")
    print(f"Loading predictions from CSV: {pred_path}")
    try:
        pred_df = pd.read_csv(pred_path, index_col=0); predictions_k = pred_df.values.astype(np.float32)
        n_test_samples_pred, n_pred_times = predictions_k.shape
        print(f"Loaded predictions for {n_test_samples_pred} samples...")
    except Exception as e: print(f"Error loading predictions CSV {pred_path}: {e}"); raise

    # --- 3. Load True Survival Data (Test Set) ---
    try:
        default_time_col = config['data']['time_column']; default_event_col = config['data']['event_column']
        clinical_col = config['data']['clinical_feature_col']; id_col = config['data'].get('id_column', None)
        t_col_test = config['data'].get('os_time_column', default_time_col)
        e_col_test = config['data'].get('os_event_column', default_event_col)

        # CHECK FOR CUSTOM OVERRIDE
        if args.custom_surv:
            test_surv_path = args.custom_surv
            test_cohort = "custom_dataset"
            print(f"\nEvaluating CUSTOM cohort from: {test_surv_path}")
        else:
            test_cohort = config['run_setup']['test_cohort']
            test_surv_path = config['data'][f"{test_cohort}_surv_path"]
            print(f"\nEvaluating Test Cohort: '{test_cohort}' (from config) at {test_surv_path}")
            
        t_test_tensor, e_test_tensor, _, _ = load_survival_data(test_surv_path, t_col_test, e_col_test, clinical_col, id_col)
        t_test_np = t_test_tensor.numpy(); e_test_np = e_test_tensor.numpy()
    except Exception as e: print(f"Error loading test survival data: {e}"); sys.exit(1)

    # --- 3b. Row Count Check ---
    if len(t_test_np) != n_test_samples_pred: 
        raise ValueError(f"CRITICAL: Row count mismatch! Predictions have {n_test_samples_pred} rows, but ground truth has {len(t_test_np)} rows.")
    else: n_test_samples = n_test_samples_pred; print(f"Verified matching row count ({n_test_samples}).")

    # --- 4. Load True Survival Data (Training Set - Needed for AUC) ---
    print("\nLoading true survival data for training cohorts (Required for dynamic AUC)...")
    t_train_list, e_train_list = [], []
    try:
        training_cohorts = config['run_setup']['training_cohorts']; t_col_train = config['data']['time_column']
        e_col_train = config['data']['event_column']; id_col_train = config['data'].get('id_column', None)
        clinical_col_train = config['data']['clinical_feature_col']
        for cohort in training_cohorts:
             train_surv_path = config['data'][f"{cohort}_surv_path"]
             t_cohort, e_cohort, _, _ = load_survival_data(train_surv_path, t_col_train, e_col_train, clinical_col_train, id_col_train)
             t_train_list.append(t_cohort); e_train_list.append(e_cohort)
        t_train_np = torch.cat(t_train_list).numpy(); e_train_np = torch.cat(e_train_list).numpy()
        print(f"Loaded combined training survival data for {len(t_train_np)} samples.")
    except Exception as e: print(f"Error loading training survival data: {e}"); sys.exit(1)

    # --- 5. Prepare Time Points ---
    try: time_bins = np.array(config['data']['time_bins'])
    except KeyError: print("Error: 'data.time_bins' not found."); sys.exit(1)
    pred_times = np.concatenate(([0.0], time_bins))
    if len(pred_times) != n_pred_times: raise ValueError("Mismatch pred times / prediction columns.")
    eval_times = np.arange(0.1, 5.01, 0.1)

    # --- 6. Interpolate Survival Probabilities ---
    print("Interpolating survival probabilities...")
    interpolated_probs = np.zeros((n_test_samples, len(eval_times)))
    for i in range(n_test_samples):
        y_interp = predictions_k[i, :]
        try: interp_func = interp1d(pred_times, y_interp, kind='linear', bounds_error=False, fill_value=(y_interp[0], y_interp[-1]))
        except ValueError as e: interpolated_probs[i, :] = np.nan; continue
        interpolated_probs[i, :] = interp_func(eval_times)
    interpolated_probs = np.clip(interpolated_probs, 0.0, 1.0)

    # --- 6b. Apply Smoothing
    window = 10
    kernel = np.ones(window) / window
    pad_left = window // 2
    pad_right = window - pad_left - 1
    smoothed_interpolated_probs = np.zeros_like(interpolated_probs)

    for i in range(n_test_samples):
        y = interpolated_probs[i, :]
        if np.isnan(y).any():
            smoothed_interpolated_probs[i, :] = np.nan
            continue
        y_padded = np.pad(y, (pad_left, pad_right), mode='reflect')
        y_conv = np.convolve(y_padded, kernel, mode='valid')
        y_mon = np.minimum.accumulate(y_conv)
        smoothed_interpolated_probs[i, :] = np.clip(y_mon, 0.0, 1.0)

    # --- 7. Calculate Concordance Index (C-index) ---
    print("\n--- Calculating C-index ---")
    eval_time_cindex = 5.0; c_index = np.nan
    idx_cindex = np.abs(eval_times - min(max(eval_times.min(), eval_time_cindex), eval_times.max())).argmin()
    
    if idx_cindex < smoothed_interpolated_probs.shape[1]:
        probs_for_cindex = smoothed_interpolated_probs[:, idx_cindex]
        valid_mask_c1 = ~np.isnan(probs_for_cindex)
        if np.any(valid_mask_c1):
            risk_scores_cindex = -probs_for_cindex[valid_mask_c1] 
            t_test_c = t_test_np[valid_mask_c1]
            e_test_c = e_test_np[valid_mask_c1]
            try: c_index = concordance_index(t_test_c, -risk_scores_cindex, e_test_c.astype(bool))
            except Exception as e: print(f"Error C-index: {e}")
            print(f"C-index: {c_index:.4f}")
        else: print("Warning: All survival probs at t~5 NaN. Cannot calc C-index.")

    # --- 8. Calculate Time-Dependent AUC  ---
    print("\n--- Calculating Time-dependent AUC ---")
    auc_mean_all = np.nan; auc_values_all = None; auc_eval_times_filtered = np.array([])
    auc_mean_target_range = np.nan
    try:
        survival_train_sks = Surv.from_arrays(event=e_train_np.astype(bool), time=t_train_np)
        survival_test_sks = Surv.from_arrays(event=e_test_np.astype(bool), time=t_test_np)

        risk_scores_time_varying = 1.0 - smoothed_interpolated_probs

        min_event_time_test = t_test_np[e_test_np == 1].min() if np.any(e_test_np == 1) else t_test_np.min()
        max_time_test = t_test_np.max(); max_time_train = t_train_np.max()
        time_mask_filtered = (eval_times >= min_event_time_test) & (eval_times < max_time_train) & (eval_times <= max_time_test)
        auc_eval_times_filtered = eval_times[time_mask_filtered] 

        if len(auc_eval_times_filtered) > 1:
            risk_scores_auc_filtered = risk_scores_time_varying[:, time_mask_filtered] 
            auc_values_all, auc_mean_all = cumulative_dynamic_auc(
                survival_train_sks, survival_test_sks, estimate=risk_scores_auc_filtered, times=auc_eval_times_filtered
            )
            print(f"Mean time-dependent AUC (data range): {auc_mean_all:.4f}")

            target_range_mask = (auc_eval_times_filtered >= 0.5) & (auc_eval_times_filtered <= 5.0)
            auc_eval_times_target = auc_eval_times_filtered[target_range_mask]

            if len(auc_eval_times_target) > 1:
                 risk_scores_auc_target = risk_scores_auc_filtered[:, target_range_mask]
                 _, auc_mean_target_range = cumulative_dynamic_auc(
                      survival_train_sks, survival_test_sks, estimate=risk_scores_auc_target, times=auc_eval_times_target
                 )
                 print(f"Mean time-dependent AUC (0.5y - 5.0y): {auc_mean_target_range:.4f}")
        else:
            print("Warning: Not enough valid evaluation times for AUC calculation.")

    except Exception as e: print(f"Error calculating time-dependent AUC: {e}")

    # --- 9. Calculate Brier Score ---
    print("\n--- Calculating Brier Score ---")
    ibs = np.nan; brier_scores_dict = {}
    try:
        brier_scores_dict = brier_score_at_times(t_test_np, smoothed_interpolated_probs, e_test_np, eval_times)
        valid_brier_scores = [s for s in brier_scores_dict.values() if not np.isnan(s)]
        if valid_brier_scores:
            area = np.trapz(valid_brier_scores, x=eval_times)
            ibs = area / (eval_times[-1] - eval_times[0])
            print(f"Integrated Brier Score: {ibs:.4f}")
    except Exception as e: print(f"Error calculating Brier score: {e}")

    # --- 10. Save or Display Results ---
    if args.output_metrics:
        results = {
            "c_index": c_index if not np.isnan(c_index) else None,
            "auc_mean_over_data_range": auc_mean_all if not np.isnan(auc_mean_all) else None,
            "auc_mean_0.5_to_5_years": auc_mean_target_range if not np.isnan(auc_mean_target_range) else None,
            "ibs": ibs if not np.isnan(ibs) else None,
            "config_file": args.config, 
            "prediction_file": pred_path, 
            "test_cohort": test_cohort
        }
        
        output_dir_metrics = os.path.dirname(args.output_metrics)
        if output_dir_metrics and not os.path.exists(output_dir_metrics): os.makedirs(output_dir_metrics)
        try:
             with open(args.output_metrics, 'w') as f:
                  json.dump(results, f, indent=4)
             print(f"\nEvaluation metrics saved to: {args.output_metrics}")
        except Exception as e: print(f"Error saving metrics to JSON: {e}")

    print("\nEvaluation script completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate survival model predictions.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predictions CSV.")
    parser.add_argument("--output_metrics", type=str, default=None, help="Path to save metrics JSON.")
    
    # NEW ARGUMENT FOR CUSTOM DATA
    parser.add_argument("--custom_surv", type=str, default=None, help="Path to custom ground-truth survival/clinical data.")

    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"\nAn error occurred during evaluation: {type(e).__name__}: {e}")
        sys.exit(1)