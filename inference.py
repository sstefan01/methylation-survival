import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import os
import sys
import glob
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
    sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during import: {e}")
     sys.exit(1)

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        required_sections = ['data', 'run_setup', 'model', 'inference']
        for section in required_sections:
            if section not in config: raise ValueError(f"Missing section '{section}'")
        return config
    except Exception as e:
        print(f"Error loading or parsing configuration file: {e}")
        raise

def main(args):
    # --- 1. Load Configuration ---
    config = load_config(args.config)
    
    ensemble_dir = args.ensemble_dir or config['inference'].get('ensemble_model_dir', config['training'].get('output_model_dir'))
    output_path = args.output or config['inference']['output_predictions_path']
    test_cohort = args.test_cohort or config['run_setup']['test_cohort']

    # --- 2. Setup Device ---
    device = torch.device("cpu") 
    print(f"Using device: {device}")

    # --- 3. Identify Features / Columns ---
    feature_types = config['run_setup']['feature_types']
    model_input_dim = config['model']['part1']['input_dim']
    id_col_name = config['data'].get('id_column', None)

    print(f"\n--- Data Setup ---")
    if args.custom_beta or args.custom_cnv:
        print("Using custom file paths for inference.")
    else:
        print(f"Running inference for configured test cohort: '{test_cohort}'")

    # --- 4. Load Connectivity Matrix ---
    conn_mat = load_connectivity_matrix(
        csv_path=config['data']['conn_mat_path'],
        out_features_s1=config['model']['part1']['layer_dims'][0],
        in_features_s1=model_input_dim,
        c2_input_offset=config['model']['part1']['c2_input_offset'],
        is_one_based=config['data']['conn_mat_is_one_based']
    ).to(device)

    # --- 5 & 6. Load and Normalize Test Data Features ---
    x_test_norm_list = []
    
    if "beta" in feature_types:
        beta_path = args.custom_beta or config['data'].get(f"{test_cohort}_beta_path")
        if not beta_path: raise ValueError("Beta path not found in config or args.")
        print(f"Loading Beta features from: {beta_path}")
        x_test_beta = load_beta_features(beta_path)
        x_test_norm_list.append(normalize_instance_wise(x_test_beta))
        
    if "cnv" in feature_types:
        cnv_path = args.custom_cnv or config['data'].get(f"{test_cohort}_cnv_path")
        if not cnv_path: raise ValueError("CNV path not found in config or args.")
        print(f"Loading CNV features from: {cnv_path}")
        x_test_cnv = load_cnv_features(cnv_path)
        x_test_norm_list.append(normalize_instance_wise(x_test_cnv))

    x_test_norm_concat = torch.cat(x_test_norm_list, dim=1)
    if x_test_norm_concat.shape[1] != model_input_dim:
        raise ValueError(f"Dimension Mismatch: Loaded features have dim {x_test_norm_concat.shape[1]}, expected {model_input_dim}")

    # --- 7. Load Clinical Data ---
    surv_path = args.custom_surv or config['data'].get(f"{test_cohort}_surv_path")
    if not surv_path: raise ValueError("Survival/Clinical path not found in config or args.")
    print(f"Loading Clinical data from: {surv_path}")
    
    t_col_test = config['data'].get('os_time_column', config['data']['time_column'])
    e_col_test = config['data'].get('os_event_column', config['data']['event_column'])
    clinical_col = config['data']['clinical_feature_col']

    _, _, m_test, patient_ids_test_loaded = load_survival_data(
        surv_path, t_col_test, e_col_test, clinical_col, id_col_name
    )
    
    

    m_test = m_test.reshape(-1, 1).float()
    val_from = config['run_setup']['metastasis_recoding']['from']
    val_to = config['run_setup']['metastasis_recoding']['to']
    clinical_col_name = config['data']['clinical_feature_col']
    print(f"Recoding '{clinical_col_name}': {val_from} -> {val_to}")
    m_test[m_test == val_from] = val_to
    print(f"Prepared metastasis shape: {m_test.shape}")

    patient_ids_test = np.asarray(patient_ids_test_loaded) if patient_ids_test_loaded is not None else None

    # Check sample alignments
    if x_test_norm_concat.shape[0] != m_test.shape[0]:
        raise ValueError(f"Sample count mismatch: Features ({x_test_norm_concat.shape[0]}) vs Clinical ({m_test.shape[0]})")

    # --- 8. Find Ensemble Models ---
    model_paths = glob.glob(os.path.join(ensemble_dir, "*_ensemble_seed_*.pt"))
    if not model_paths:
        raise FileNotFoundError(f"No ensemble models found in directory: {ensemble_dir}")
    print(f"\nFound {len(model_paths)} models in ensemble directory.")

    # --- 9. Instantiate Base Model Architecture ---
    # CHANGES HERE: Updated instantiation for the Embedding table parameters
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

    # --- 10. Prepare DataLoader ---
    batch_size = config['inference']['batch_size']
    test_dataset = TensorDataset(x_test_norm_concat.to(device), m_test.to(device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --- 11. Run Inference Loop for EACH Model ---
    all_ensemble_preds = [] 
    
    print("\n--- Starting Ensemble Inference ---")
    with torch.no_grad():
        for m_idx, path in enumerate(model_paths):
            print(f"Predicting with model {m_idx+1}/{len(model_paths)}: {os.path.basename(path)}")
            
            model.load_state_dict(torch.load(path, map_location=device), strict=False)
            model.eval()
            
            model_preds = []
            for batch_x_main, batch_x_clinical in test_loader:

                logits = model(batch_x_main, batch_x_clinical)
                survival_probs = mtlr_survival(logits)
                model_preds.append(survival_probs.cpu().numpy())
                
            all_ensemble_preds.append(np.concatenate(model_preds, axis=0))

    # --- 12. Aggregate Ensemble Predictions ---
    print("\nAggregating predictions across all models...")
    ensemble_stack = np.stack(all_ensemble_preds, axis=0)
    final_predictions = np.mean(ensemble_stack, axis=0) 

    # --- 13. Save Output to CSV ---
    time_bins = np.array(config['data']['time_bins'])
    pred_times = np.concatenate(([0.0], time_bins))
    time_headers = [f"S_t_{t:.4f}" for t in pred_times]

    pred_df = pd.DataFrame(final_predictions, columns=time_headers)
    index_flag = True

    if patient_ids_test is not None and id_col_name:
        pred_df.insert(0, id_col_name, patient_ids_test)
        index_flag = False

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)

    print(f"\nSaving final averaged predictions to: {output_path}")
    pred_df.to_csv(output_path, index=index_flag, float_format='%.8f')
    print("Success!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",type=str,default="config.yaml",help="Path to YAML config.")
    parser.add_argument("--ensemble_dir", type=str, help="Directory containing the ensemble .pt files.")
    parser.add_argument("--output", type=str, help="Override path to save output predictions (.csv).")
    parser.add_argument("--test_cohort", type=str, help="Override the test cohort.")
    
    # New Custom Data Arguments
    parser.add_argument("--custom_beta", type=str, help="Path to custom beta features file.")
    parser.add_argument("--custom_cnv", type=str, help="Path to custom CNV features file.")
    parser.add_argument("--custom_surv", type=str, help="Path to custom clinical/survival data file.")
    
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"\nAn error occurred during inference: {type(e).__name__}: {e}")
        sys.exit(1)