# train.py (Corrected Call to load_connectivity_matrix)

import argparse
import yaml
import torch
import numpy as np
import os
import sys
import random
import torch.optim as optim
import warnings # For suppressing Dask warnings if needed

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# --- Project Imports ---
try:
    from model import CombinedSurvivalModel
    from utils import (
        load_connectivity_matrix,
        load_beta_features,
        load_cnv_features,
        load_survival_data, # Assumes returns t, e, m, ids (torch tensors/np array)
        normalize_instance_wise, # Instance-wise normalizer
        encode_survival, # Target encoder
        mtlr_neg_log_likelihood # Loss function
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"Ensure src directory '{src_dir}' is accessible and contains model.py/utils.py")
    sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during project import: {e}")
     sys.exit(1)

# --- Config Loader ---
def load_config(config_path):
    """Loads the YAML configuration file with basic validation."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
        # Basic validation
        required_sections = ['data', 'run_setup', 'model', 'training']
        for section in required_sections:
            if section not in config: raise ValueError(f"Missing required section '{section}'")
        required_data_keys = ['time_column', 'event_column', 'clinical_feature_col', 'time_bins', 'conn_mat_path']
        for key in required_data_keys:
             if key not in config['data']: raise ValueError(f"Missing required key 'data.{key}'")
        required_run_setup_keys = ['training_cohorts', 'feature_types']
        for key in required_run_setup_keys:
             if key not in config['run_setup']: raise ValueError(f"Missing required key 'run_setup.{key}'")
        required_training_keys = ['epochs', 'batch_size', 'learning_rate', 'weight_decay', 'optimizer', 'output_model_dir', 'output_model_name']
        for key in required_training_keys:
             if key not in config['training']: raise ValueError(f"Missing required key 'training.{key}'")
        return config
    except Exception as e:
        print(f"Error loading or parsing configuration file {config_path}: {e}")
        raise


# --- Main Training Function ---
def main(args):
    """Runs the training process."""

    # --- 1. Load Configuration & Basic Setup ---
    config = load_config(args.config)
    config['training']['epochs'] = args.epochs or config['training']['epochs']

    # --- 2. Setup Device & Seed ---
    device = torch.device("cpu") # Force CPU
    print(f"Using device: {device}")
    seed = 42 ; torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    print(f"Set random seed: {seed}")

    # --- 3. Load & Prepare Training Data ---
    print("Loading and preparing training data...")
    try: # Load data and prepare x_train, m_train, y_train
        # ... (get config values: cohorts, features, columns, bins, etc.) ...
        training_cohorts = config['run_setup']['training_cohorts']; feature_types = config['run_setup']['feature_types']
        time_col = config['data']['time_column']; event_col = config['data']['event_column']
        clinical_col = config['data']['clinical_feature_col']; id_col = config['data'].get('id_column', None)
        time_bins = torch.tensor(config['data']['time_bins'], dtype=torch.float32).to(device) # Load bins to device
        recoding_rule = config['run_setup']['metastasis_recoding']; model_input_dim = config['model']['part1']['input_dim']

        # ... (load features cohort by cohort into feature_data dict) ...
        feature_data = {ftype: [] for ftype in feature_types}; survival_data = {'t': [], 'e': [], 'm': []}
        print(f"Loading data for training cohorts: {training_cohorts}")
        for cohort in training_cohorts:
             print(f"  Loading cohort: {cohort}")
             for ftype in feature_types: # Load features
                  path_key = f"{cohort}_{ftype}_path"; path = config['data'][path_key]
                  if ftype == "beta": feature_data[ftype].append(load_beta_features(path))
                  elif ftype == "cnv": feature_data[ftype].append(load_cnv_features(path))
                  else: raise ValueError(f"Unknown feature type '{ftype}'")
             surv_path_key = f"{cohort}_surv_path"; surv_path = config['data'][surv_path_key] # Load survival
             t_cohort, e_cohort, m_cohort, _ = load_survival_data(surv_path, time_col, event_col, clinical_col, id_col)
             survival_data['t'].append(t_cohort); survival_data['e'].append(e_cohort); survival_data['m'].append(m_cohort)

        # ... (concatenate survival data: all_train_t, all_train_e, all_train_m) ...
        all_train_t = torch.cat(survival_data['t']); all_train_e = torch.cat(survival_data['e']); all_train_m = torch.cat(survival_data['m'])

        # ... (normalize features separately then concatenate -> x_train) ...
        print("Normalizing and concatenating features...")
        norm_features_list = []; expected_samples = len(all_train_t)
        for ftype in feature_types:
             if not feature_data[ftype]: continue
             x_train_type = torch.cat(feature_data[ftype], dim=0)
             if x_train_type.shape[0] != expected_samples: raise ValueError(f"Sample count mismatch '{ftype}'")
             print(f"  Normalizing {ftype.upper()} features (shape: {x_train_type.shape})...")
             x_train_type_norm = normalize_instance_wise(x_train_type); norm_features_list.append(x_train_type_norm)
        if not norm_features_list: raise ValueError("No feature data.")
        x_train = torch.cat(norm_features_list, dim=1)
        print(f"Final training features shape: {x_train.shape}")
        if x_train.shape[1] != model_input_dim: raise ValueError("Final feature dim mismatch.")

        # ... (prepare m_train, y_train) ...
        m_train = all_train_m.reshape(-1, 1)
        val_from = recoding_rule['from']; val_to = recoding_rule['to']; m_train[m_train == val_from] = val_to
        y_train = encode_survival(all_train_t, all_train_e, time_bins)
        print(f"Training data prepared: X={x_train.shape}, M={m_train.shape}, Y={y_train.shape}")
        n_samples = y_train.size(dim=0)

    except Exception as e: print(f"Error during data loading/preprocessing: {e}"); sys.exit(1)


    # --- 4. Load Connectivity Matrix ---
    print("Loading connectivity matrix...")
    try:
        conn_mat_path = config['data']['conn_mat_path']
        # Get features from config (ensure model section is loaded and keys exist)
        out_features = config['model']['part1']['layer_dims'][0]
        in_features = config['model']['part1']['input_dim']
        offset = config['model']['part1']['c2_input_offset']
        one_based = config['data']['conn_mat_is_one_based']

        conn_mat = load_connectivity_matrix(
            csv_path=conn_mat_path,
            out_features_s1=out_features, # Pass value
            in_features_s1=in_features,   # Pass value
            c2_input_offset=offset,       # Pass value
            is_one_based=one_based        # Pass value
        ).to(device) # Move to device
        print("Connectivity matrix loaded.")
    except KeyError as e:
        print(f"Error: Missing required key {e} in config file needed for loading connectivity matrix.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading connectivity matrix: {e}")
        sys.exit(1)


    # --- 5. Instantiate Model ---
    print("Instantiating model...")
    try:
        model = CombinedSurvivalModel(
            part1_input_dim=config['model']['part1']['input_dim'],
            conn_mat=conn_mat, # Pass the loaded conn_mat
            part1_layer_dims=config['model']['part1']['layer_dims'],
            part1_dropout_rate=config['model']['part1']['dropout_rate'],
            num_clinical_features=config['model']['combined']['num_clinical_features'],
            clinical_feature_weight=config['model']['combined']['clinical_feature_weight'],
            part2_num_time_bins=config['model']['part2']['num_time_bins'],
            part2_dropout_rate=config['model']['part2']['dropout_rate']
        ).to(device)
    except Exception as e: print(f"Error instantiating model: {e}"); sys.exit(1)


    # --- 6. Setup Optimizer ---
    try:
         lr = config['training']['learning_rate']; wd = config['training']['weight_decay']; opt_name = config['training']['optimizer']
         if opt_name.lower() == "adam": optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
         else: raise ValueError(f"Unsupported optimizer: {opt_name}")
         print(f"Using Optimizer: {opt_name}, LR: {lr}, Weight Decay: {wd}")
    except Exception as e: print(f"Error setting up optimizer: {e}"); sys.exit(1)


    # --- 7. Training Loop ---
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    print(f"\n--- Starting Training for {epochs} Epochs ---")

    try:
        x_train = x_train.to(device)
        m_train = m_train.to(device)
        y_train = y_train.to(device)
        print("Moved training data tensors to device.")
    except Exception as e: print(f"Error moving data to device {device}: {e}. Check memory."); sys.exit(1)
    model.train()
    for epoch in range(epochs):
        loss = 0
        for bit in range(0,round(y_train.size(dim=0)/batch_size)-1):
            batch_features = x_train[bit*batch_size:(1+bit)*batch_size,]
            met_features = m_train[bit*batch_size:(1+bit)*batch_size,]
            lab_features = y_train[bit*batch_size:(1+bit)*batch_size,]
            optimizer.zero_grad()
            out = model(batch_features.float(), met_features.float())
            l = mtlr_neg_log_likelihood(out, lab_features, average=True)
            l.backward()
            optimizer.step()
            loss += l.item()

    print("--- Training Finished ---")

    # --- 8. Save Model State Dictionary ---
    output_dir = config['training']['output_model_dir']; output_name = config['training']['output_model_name']
    if "statedict" not in output_name.lower(): # Append _statedict if needed
         base, ext = os.path.splitext(output_name)
         output_name = f"{base}_statedict{ext if ext else '.pt'}"
    save_path = os.path.join(output_dir, output_name)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f"Saving trained model state_dict to: {save_path}")
    try: torch.save(model.state_dict(), save_path); print("Model state_dict saved successfully.")
    except Exception as e: print(f"Error saving model state_dict: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the survival model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",type=str,default="config.yaml",help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, help="Override number of training epochs.")
    args = parser.parse_args()
    try: main(args)
    except (FileNotFoundError, ValueError, KeyError, Exception) as e:
        print(f"\nAn error occurred during training: {type(e).__name__}: {e}")
        sys.exit(1)