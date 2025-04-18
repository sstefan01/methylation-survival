# inference.py

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import os
import sys
from torch.utils.data import TensorDataset, DataLoader

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Project and third-party imports ---
try:
    from model import CombinedSurvivalModel
    from utils import (
        load_connectivity_matrix,
        load_beta_features,
        load_cnv_features,
        load_survival_data,
        normalize_instance_wise,
        mtlr_survival
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"Ensure src directory '{src_dir}' is accessible and contains model.py/utils.py")
    sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during import: {e}")
     sys.exit(1)

def load_config(config_path):
    """Loads the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
        required_sections = ['data', 'run_setup', 'model', 'inference']
        for section in required_sections:
            if section not in config: raise ValueError(f"Missing section '{section}'")
        if 'id_column' not in config['data']: print("Warning: 'data.id_column' not in config. Output CSV will lack Patient IDs.")
        return config
    except Exception as e:
        print(f"Error loading or parsing configuration file {config_path}: {e}")
        raise

def main(args):
    """Runs the inference process."""

    # --- 1. Load Configuration ---
    config = load_config(args.config)
    # Override paths/settings if command-line arguments are provided
    config['inference']['input_model_path'] = args.model_path or config['inference']['input_model_path']
    config['inference']['output_predictions_path'] = args.output or config['inference']['output_predictions_path']
    config['run_setup']['test_cohort'] = args.test_cohort or config['run_setup']['test_cohort']

    # --- 2. Setup Device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Identify Test Cohort / Features / Columns ---
    try:
        test_cohort = config['run_setup']['test_cohort']
        feature_types = config['run_setup']['feature_types']
        clinical_col_name = config['data']['clinical_feature_col']
        recoding_rule = config['run_setup']['metastasis_recoding']
        model_input_dim = config['model']['part1']['input_dim']
        id_col_name = config['data'].get('id_column', None)
    except KeyError as e:
        print(f"Error: Missing required key {e} in config file.")
        sys.exit(1)

    print(f"Running inference for test cohort: '{test_cohort}'")
    print(f"Using feature types: {feature_types}")
    if id_col_name: print(f"Expecting patient ID column: '{id_col_name}'")

    # --- 4. Load Connectivity Matrix ---
    try:
        conn_mat = load_connectivity_matrix(
            csv_path=config['data']['conn_mat_path'],
            out_features_s1=config['model']['part1']['layer_dims'][0],
            in_features_s1=model_input_dim,
            c2_input_offset=config['model']['part1']['c2_input_offset'],
            is_one_based=config['data']['conn_mat_is_one_based']
        ).to(device)
    except Exception as e:
        print(f"Error loading connectivity matrix: {e}")
        sys.exit(1)

    # --- 5. Load Test Data Features ---
    print(f"Loading features for test cohort '{test_cohort}'...")
    x_test_beta = None
    x_test_cnv = None

    try:
        if "beta" in feature_types:
            beta_path_key = f"{test_cohort}_beta_path"
            if beta_path_key not in config['data']: raise ValueError(f"Path key '{beta_path_key}' not found")
            x_test_beta = load_beta_features(config['data'][beta_path_key])
        if "cnv" in feature_types:
            cnv_path_key = f"{test_cohort}_cnv_path"
            if cnv_path_key not in config['data']: raise ValueError(f"Path key '{cnv_path_key}' not found")
            x_test_cnv = load_cnv_features(config['data'][cnv_path_key])

    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"Error loading test features: {e}")
        sys.exit(1)

    # --- 6. Apply Instance-wise Normalization ---
    x_test_norm_list = [] # List to hold normalized feature tensors

    if x_test_beta is not None:
        print(f"Applying instance-wise normalization to BETA features (shape: {x_test_beta.shape})...")
        x_test_beta_norm = normalize_instance_wise(x_test_beta)
        x_test_norm_list.append(x_test_beta_norm)
    if x_test_cnv is not None:
        print(f"Applying instance-wise normalization to CNV features (shape: {x_test_cnv.shape})...")
        x_test_cnv_norm = normalize_instance_wise(x_test_cnv)
        x_test_norm_list.append(x_test_cnv_norm)

    if not x_test_norm_list:
        raise ValueError("No features were loaded or normalized.")

    # --- 7. Concatenate Normalized Features ---
    x_test_norm_concat = torch.cat(x_test_norm_list, dim=1)
    print(f"Concatenated normalized test features shape: {x_test_norm_concat.shape}")

    # Check if final dimension matches model input expectation
    if x_test_norm_concat.shape[1] != model_input_dim:
        raise ValueError(f"Dimension Mismatch: Final normalized features dim ({x_test_norm_concat.shape[1]}) "
                         f"!= model input dim ({model_input_dim})")

    # --- 8. Load Clinical Feature and Patient IDs ---
    print(f"Loading clinical data and IDs for test cohort '{test_cohort}'...")
    patient_ids_test = None # Initialize
    try:
        surv_path_key = f"{test_cohort}_surv_path"
        if surv_path_key not in config['data']: raise ValueError(f"Path key '{surv_path_key}' not found")
        test_surv_path = config['data'][surv_path_key]

        t_col_test = config['data'].get('os_time_column', config['data']['time_column'])
        e_col_test = config['data'].get('os_event_column', config['data']['event_column'])
        clinical_col = config['data']['clinical_feature_col']

        _, _, m_test, patient_ids_test_loaded = load_survival_data(
            test_surv_path, t_col_test, e_col_test, clinical_col, id_col_name # Pass ID col name
        )
        if id_col_name and patient_ids_test_loaded is not None:
             patient_ids_test = np.asarray(patient_ids_test_loaded)
             print(f"Loaded IDs for {len(patient_ids_test)} test samples.")
        elif id_col_name and patient_ids_test_loaded is None:
             print(f"Warning: Requested ID column '{id_col_name}' but it was not found or returned.")

    except (ValueError, FileNotFoundError, Exception) as e:
         print(f"Error loading clinical/survival data: {e}")
         sys.exit(1)

    # Reshape and recode metastasis
    m_test = m_test.reshape(-1, 1)
    val_from = config['run_setup']['metastasis_recoding']['from']
    val_to = config['run_setup']['metastasis_recoding']['to']
    print(f"Recoding '{clinical_col_name}': {val_from} -> {val_to}")
    m_test[m_test == val_from] = val_to
    print(f"Prepared metastasis shape: {m_test.shape}")

    # Check sample counts
    n_samples = x_test_norm_concat.shape[0]
    if n_samples != m_test.shape[0]:
        raise ValueError(f"Sample count mismatch: features ({n_samples}) vs metastasis ({m_test.shape[0]})")
    if patient_ids_test is not None and n_samples != len(patient_ids_test):
        raise ValueError(f"Sample count mismatch: features ({n_samples}) vs patient ids ({len(patient_ids_test)})")

    # --- 9. Instantiate Model ---
    print("Instantiating model structure...")
    try:
        model = CombinedSurvivalModel(
            part1_input_dim=config['model']['part1']['input_dim'],
            conn_mat=conn_mat,
            part1_layer_dims=config['model']['part1']['layer_dims'],
            part1_dropout_rate=config['model']['part1']['dropout_rate'],
            num_clinical_features=config['model']['combined']['num_clinical_features'],
            clinical_feature_weight=config['model']['combined']['clinical_feature_weight'],
            part2_num_time_bins=config['model']['part2']['num_time_bins'],
            part2_dropout_rate=config['model']['part2']['dropout_rate']
        ).to(device)
    except Exception as e:
        print(f"Error instantiating model: {e}")
        sys.exit(1)

    # --- 10. Load Pre-trained Weights (State Dictionary) ---
    model_path = config['inference']['input_model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model state_dict file not found: {model_path}")
    print(f"Loading model state_dict from: {model_path}")
    try:
        state_dict = torch.load(model_path, map_location=device)
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"Model state_dict loaded. Missing keys: {load_result.missing_keys}. Unexpected keys: {load_result.unexpected_keys}")
        if load_result.unexpected_keys:
             print("Warning: Model loaded with unexpected keys in state_dict. Check compatibility.")
        missing_non_buffer = [k for k in load_result.missing_keys if 'clinical_weight' not in k] # Example check
        if missing_non_buffer:
             print(f"Warning: Model loaded with missing keys potentially affecting weights: {missing_non_buffer}")

    except Exception as e:
        print(f"Error loading state_dict: {e}")
        raise

    # --- 11. Prepare DataLoader ---
    batch_size = config['inference']['batch_size']
    x_test_final = x_test_norm_concat.to(device)
    m_test = m_test.to(device)
    test_dataset = TensorDataset(x_test_final, m_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Using batch size: {batch_size} for inference.")

    # --- 12. Run Inference Loop ---
    model.eval()
    all_predictions = []
    print("Starting inference loop...")
    with torch.no_grad():
        for i, (batch_x_main, batch_x_clinical) in enumerate(test_loader):
            logits = model(batch_x_main, batch_x_clinical)
            survival_probs = mtlr_survival(logits)
            all_predictions.append(survival_probs.cpu().numpy())
            if (i + 1) % 50 == 0 or i == len(test_loader) - 1:
                 print(f"  Processed batch {i+1}/{len(test_loader)}")
    print("Inference loop finished.")

    # --- 13. Combine Predictions and Prepare for CSV ---
    print("Combining batch predictions...")
    final_predictions = np.concatenate(all_predictions, axis=0)

    # Generate Time Point Headers
    try:
        time_bins = np.array(config['data']['time_bins'])
        pred_times = np.concatenate(([0.0], time_bins))
        if len(pred_times) != final_predictions.shape[1]: raise ValueError("Mismatch time_bins / prediction columns.")
        time_headers = [f"S_t_{t:.4f}" for t in pred_times]
    except (KeyError, ValueError) as e:
         print(f"Warning: Error generating time headers ({e}). Using generic headers.")
         time_headers = [f"Prob_Time_{i}" for i in range(final_predictions.shape[1])]

    # Create Pandas DataFrame
    pred_df = pd.DataFrame(final_predictions, columns=time_headers)

    # Add Patient IDs if available
    if patient_ids_test is not None and id_col_name:
        pred_df.insert(0, id_col_name, patient_ids_test)
        index_flag = False
    else:
        index_flag = True # Write default pandas index if IDs missing or column name unknown

    # --- 14. Save Predictions as CSV ---
    output_path = config['inference']['output_predictions_path']
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    print(f"Saving predictions DataFrame shape {pred_df.shape} to CSV: {output_path}")
    try:
        pred_df.to_csv(output_path, index=index_flag, float_format='%.8f')
        print("Predictions saved successfully as CSV.")
    except Exception as e:
        print(f"Error saving predictions to CSV: {e}")
        npy_fallback_path = os.path.splitext(output_path)[0] + "_fallback.npy"
        try:
             print(f"Attempting fallback save to NPY: {npy_fallback_path}")
             np.save(npy_fallback_path, final_predictions)
        except Exception as e_npy:
             print(f"Fallback NPY save also failed: {e_npy}")


    print("Inference script completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference using the pre-trained survival model based on configuration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config",type=str,default="config.yaml",help="Path to the YAML configuration file.")
    parser.add_argument("--model_path", type=str, help="Override path to pre-trained model state_dict (.pt) file.")
    parser.add_argument("--output", type=str, help="Override path to save output predictions (.csv).")
    parser.add_argument("--test_cohort", type=str, help="Override the test cohort specified in the config's run_setup section.")

    args = parser.parse_args()
    try:
        main(args)
    except (FileNotFoundError, ValueError, KeyError, Exception) as e:
        print(f"\nAn error occurred during inference: {type(e).__name__}: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for full traceback during debugging
        sys.exit(1)