# plot_results.py

import argparse
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from typing import List, Optional, Tuple
from scipy.interpolate import interp1d

def load_config(config_path):
    """Loads the YAML configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
        # Basic validation for keys needed by this script
        if 'data' not in config or 'time_bins' not in config['data']:
            raise ValueError("Config missing 'data' section or 'data.time_bins' key.")
        if 'inference' not in config or 'output_predictions_path' not in config['inference']:
             # Allow overriding via args, but warn if default needed and missing
             pass # Let later code handle missing prediction path if needed
        return config
    except Exception as e:
        print(f"Error loading or parsing configuration file {config_path}: {e}")
        raise



def plot_survival_curves(predictions: np.ndarray, pred_times: np.ndarray, output_path: str, num_curves_to_plot: int = 100, plot_time_max: float = 5.0, plot_time_points: int = 50, smoothing_window: int = 10):
    n_samples, n_times = predictions.shape
    if len(pred_times) != n_times: raise ValueError(f"Mismatch pred times ({n_times}) vs pred_times array ({len(pred_times)}).")
    if smoothing_window < 1: smoothing_window = 1
    plot_times = np.linspace(0, plot_time_max, plot_time_points)
    plt.figure(figsize=(8, 8))
    if n_samples > num_curves_to_plot: indices = np.random.choice(n_samples, num_curves_to_plot, replace=False); msg = "random"
    else: indices = np.arange(n_samples); msg = "all"
    for i in indices:
        sample_prediction_k = predictions[i, :]
        interp_func = interp1d(pred_times, sample_prediction_k, kind='linear', bounds_error=False, fill_value=(1.0, sample_prediction_k[-1]))
        sample_survival_interp = np.clip(interp_func(plot_times), 0, 1)
        if smoothing_window > 1:
            current_window = min(smoothing_window, len(sample_survival_interp))
            if current_window % 2 == 0: current_window -= 1
            if current_window > 1:
                pad_width = (current_window - 1) // 2; kernel = np.ones(current_window) / current_window
                padded_signal = np.pad(sample_survival_interp, pad_width, mode='edge')
                smoothed_curve = np.convolve(padded_signal, kernel, mode='valid')
                for j in range(1, len(smoothed_curve)): smoothed_curve[j] = min(smoothed_curve[j], smoothed_curve[j-1])
            else: smoothed_curve = sample_survival_interp
        else: smoothed_curve = sample_survival_interp
        if len(smoothed_curve) > 0: smoothed_curve[0] = 1.0
        smoothed_curve = np.clip(smoothed_curve, 0.0, 1.0)
        plt.plot(plot_times, smoothed_curve, alpha=0.4, linewidth=0.9)
    plt.xlabel("Time (Years)"); plt.ylabel("Survival Probability S(t)")
    plt.title("Predicted Survival Curves")
    plt.ylim(0, 1.05); plt.xlim(0, plot_time_max); plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(); plt.savefig(output_path); plt.close()
    print(f"survival curve plot saved to: {output_path}")

def plot_metric_curve(times: List[float], metric_values: List[float], metric_name: str, output_path: str, y_label: str, y_lim: Optional[Tuple[float, float]] = None):
    if not times or not metric_values: print(f"Warning: No data for {metric_name}. Skipping plot."); return
    if len(times) != len(metric_values): print(f"Warning: Mismatch len times/values for {metric_name}. Skipping plot."); return
    plt.figure(figsize=(10, 6)); plt.plot(times, metric_values, marker='.', linestyle='-', markersize=4)
    plt.xlabel("Time (Years)"); plt.ylabel(y_label); plt.title(f"Time-dependent {metric_name}")
    if y_lim: plt.ylim(y_lim)
    else:
        valid_values = [v for v in metric_values if not np.isnan(v)]
        if valid_values:
            min_val, max_val = np.min(valid_values), np.max(valid_values)
            padding = (max_val - min_val) * 0.05 if max_val > min_val else 0.1
            plt.ylim(min_val - padding, max_val + padding)
    plt.xlim(left=0); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout(); plt.savefig(output_path); plt.close()
    print(f"{metric_name} plot saved to: {output_path}")

def main(args):
    # --- 1. Load Config ---
    config = load_config(args.config)

    # --- 2. Create Output Directory ---
    if not os.path.exists(args.output_dir):
        print(f"Creating output directory: {args.output_dir}")
        os.makedirs(args.output_dir)

    # --- 3. Plot Metrics (Requires Metrics File) ---
    metrics = None
    if args.metrics_file:
        if not os.path.exists(args.metrics_file):
            print(f"Warning: Metrics file not found: {args.metrics_file}. Skipping metric plots.")
        else:
            print(f"Loading metrics from: {args.metrics_file}")
            try:
                with open(args.metrics_file, 'r') as f:
                    metrics = json.load(f)
                print("Metrics loaded successfully.")
            except Exception as e:
                print(f"Error loading metrics JSON {args.metrics_file}: {e}. Skipping metric plots.")
                metrics = None # Ensure metrics is None if loading fails

    if metrics:
        # Plot AUC Curve
        if metrics.get('auc_times_evaluated') and metrics.get('auc_values_at_evaluated_times'):
            print("Plotting AUC curve...")
            plot_metric_curve(
                times=metrics['auc_times_evaluated'], metric_values=metrics['auc_values_at_evaluated_times'],
                metric_name="Cumulative/Dynamic AUC", output_path=os.path.join(args.output_dir, "auc_curve.png"),
                y_label="AUC", y_lim=(0.0, 1.0)
            )
        else: print("Skipping AUC plot (data not found in metrics file).")

        # Plot Brier Score Curve
        if metrics.get('brier_score_times') and metrics.get('brier_score_values'):
            print("Plotting Brier Score curve...")
            brier_times = np.array(metrics['brier_score_times'])
            brier_values = np.array(metrics['brier_score_values'])
            valid_indices = ~np.isnan(brier_values)
            if np.any(valid_indices):
                 plot_metric_curve(
                     times=brier_times[valid_indices].tolist(), metric_values=brier_values[valid_indices].tolist(),
                     metric_name="Brier Score", output_path=os.path.join(args.output_dir, "brier_score_curve.png"),
                     y_label="Brier Score", y_lim=(0.0, None)
                 )
            else: print("Skipping Brier score plot (no valid score data found).")
        else: print("Skipping Brier Score plot (data not found in metrics file).")
    else:
        print("No metrics file provided or loaded. Skipping AUC and Brier plots.")


    # --- 4. Plot Survival Curves ---
    if args.plot_survival:
        # Determine prediction file path
        pred_path = args.predictions_file or config['inference'].get('output_predictions_path')
        if not pred_path:
            print("Error: Cannot plot survival curves. Need --predictions_file argument or 'inference.output_predictions_path' in config.")
        elif not os.path.exists(pred_path):
            print(f"Error: Prediction file not found at '{pred_path}'. Cannot plot survival curves.")
        else:
            print(f"Loading predictions from {pred_path} for survival curve plot...")
            try:
                # Load predictions (assuming CSV with ID as index col 0)
                pred_df = pd.read_csv(pred_path, index_col=0)
                predictions = pred_df.values.astype(np.float32)
                n_samples, n_pred_times_loaded = predictions.shape

                # Get pred_times from config
                time_bins = np.array(config['data']['time_bins'])
                pred_times = np.concatenate(([0.0], time_bins))
                if len(pred_times) != n_pred_times_loaded:
                     raise ValueError(f"Mismatch between time bins in config ({len(pred_times)} points) and prediction columns ({n_pred_times_loaded}).")

                print("Plotting individual survival curves...")
                plot_survival_curves(
                    predictions=predictions,
                    pred_times=pred_times,
                    output_path=os.path.join(args.output_dir, "survival_curves.png")
                )
            except KeyError:
                 print("Error: Cannot plot survival curves. Missing 'data.time_bins' in config.")
            except ValueError as e:
                 print(f"Error plotting survival curves: {e}")
            except Exception as e:
                print(f"Error loading predictions or plotting survival curves: {e}")
    else:
        print("Skipping survival curve plot (--plot_survival not specified).")

    print(f"\nPlots saved in directory (if generated): {args.output_dir}")
    print("Plotting script completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot evaluation metrics and survival curves from model results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (needed for time_bins if plotting survival)."
    )
    parser.add_argument(
        "--metrics_file",
        type=str,
        # Removed required=True - now optional
        help="Path to the JSON file containing evaluation metrics (output from evaluate.py)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/plots",
        help="Directory to save the generated plots."
    )
    parser.add_argument(
        "--plot_survival",
        action="store_true",
        help="Flag to generate the plot of individual survival curves."
    )
    parser.add_argument(
        "--predictions_file",
        type=str,
        default=None,
        help="Path to the prediction file (.csv). Needed if --plot_survival is used and path not in config."
    )

    args = parser.parse_args()
    try:
        main(args)
    except (FileNotFoundError, ValueError, KeyError, Exception) as e:
        print(f"\nAn error occurred during plotting: {type(e).__name__}: {e}")
        sys.exit(1)
