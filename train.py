import argparse
import yaml
import torch
import numpy as np
import os
import sys
import random
import torch.optim as optim
import warnings

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
        load_survival_data,
        normalize_instance_wise,
        encode_survival,
        mtlr_neg_log_likelihood
    )
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during project import: {e}")
     sys.exit(1)

# --- Config Loader ---
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config {config_path}: {e}")
        raise

def set_seed(seed):
    """Sets the seed for reproducibility within a specific ensemble run."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# --- Main Training Function ---
def main(args):
    # --- 1. Load Configuration & Setup ---
    config = load_config(args.config)
    config['training']['epochs'] = args.epochs or config['training']['epochs']
    # device = torch.device("cpu") 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load & Prepare Training Data (Done ONCE) ---
    print("Loading and preparing training data...")
    try: 
        training_cohorts = config['run_setup']['training_cohorts']
        feature_types = config['run_setup']['feature_types']
        time_col = config['data']['time_column']
        event_col = config['data']['event_column']
        clinical_col = config['data']['clinical_feature_col']
        recoding_rule = config['run_setup']['metastasis_recoding']; model_input_dim = config['model']['part1']['input_dim']
        id_col = config['data'].get('id_column', None)
        time_bins = torch.tensor(config['data']['time_bins'], dtype=torch.float32).to(device)
        model_input_dim = config['model']['part1']['input_dim']

        feature_data = {ftype: [] for ftype in feature_types}
        survival_data = {'t': [], 'e': [], 'm': []}
        
        print(f"Loading data for training cohorts: {training_cohorts}")
        for cohort in training_cohorts:
             for ftype in feature_types:
                  path_key = f"{cohort}_{ftype}_path"; path = config['data'][path_key]
                  if ftype == "beta": feature_data[ftype].append(load_beta_features(path))
                  elif ftype == "cnv": feature_data[ftype].append(load_cnv_features(path))
                  else: raise ValueError(f"Unknown feature type '{ftype}'")
             surv_path_key = f"{cohort}_surv_path"; surv_path = config['data'][surv_path_key]
             t_cohort, e_cohort, m_cohort, _ = load_survival_data(surv_path, time_col, event_col, clinical_col, id_col)
             survival_data['t'].append(t_cohort); survival_data['e'].append(e_cohort); survival_data['m'].append(m_cohort)

        all_train_t = torch.cat(survival_data['t'])
        all_train_e = torch.cat(survival_data['e'])
        all_train_m = torch.cat(survival_data['m'])

        print("Normalizing and concatenating features...")
        norm_features_list = []; expected_samples = len(all_train_t)
        for ftype in feature_types:
             if not feature_data[ftype]: continue
             x_train_type = torch.cat(feature_data[ftype], dim=0)
             if x_train_type.shape[0] != expected_samples: raise ValueError(f"Sample count mismatch '{ftype}'")
             x_train_type_norm = normalize_instance_wise(x_train_type)
             norm_features_list.append(x_train_type_norm)
        
        x_train = torch.cat(norm_features_list, dim=1)
        if x_train.shape[1] != model_input_dim: raise ValueError("Final feature dim mismatch.")


        m_train = all_train_m.reshape(-1, 1).float()
        val_from = recoding_rule['from']; val_to = recoding_rule['to']; m_train[m_train == val_from] = val_to
        y_train = encode_survival(all_train_t, all_train_e, time_bins)
        print(f"Training data prepared: X={x_train.shape}, M={m_train.shape}, Y={y_train.shape}")
        n_samples = y_train.size(dim=0)
        
        print(f"Training data prepared: X={x_train.shape}, M={m_train.shape}, Y={y_train.shape}")
        
    except Exception as e: 
        print(f"Error during data loading/preprocessing: {e}")
        sys.exit(1)

    # --- 3. Load Connectivity Matrix (Done ONCE) ---
    print("Loading connectivity matrix...")
    try:
        conn_mat = load_connectivity_matrix(
            csv_path=config['data']['conn_mat_path'],
            out_features_s1=config['model']['part1']['layer_dims'][0],
            in_features_s1=config['model']['part1']['input_dim'],
            c2_input_offset=config['model']['part1']['c2_input_offset'],
            is_one_based=config['data']['conn_mat_is_one_based']
        ).to(device)
    except Exception as e:
        print(f"Error loading connectivity matrix: {e}"); sys.exit(1)

    # --- 4. Deep Ensemble Setup ---
    ensemble_seeds = [42, 123, 7, 99, 2026]
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    wd = config['training']['weight_decay']
    opt_name = config['training']['optimizer']
    output_dir = config['training']['output_model_dir']
    
    if output_dir and not os.path.exists(output_dir): 
        os.makedirs(output_dir)

    print(f"\n--- Starting Deep Ensemble Training ({len(ensemble_seeds)} Models) ---")

    # --- 5. Ensemble Training Loop ---
    for i, seed in enumerate(ensemble_seeds):
        print(f"\n>> Training Model {i+1}/{len(ensemble_seeds)} (Seed: {seed}) <<")
        
        set_seed(seed)
        try:
            # CHANGES HERE: Updated instantiation to match the new Embedding arguments
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
                
            if opt_name.lower() == "adam": 
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            else: 
                raise ValueError(f"Unsupported optimizer: {opt_name}")
        except Exception as e: 
            print(f"Error instantiating model for seed {seed}: {e}"); sys.exit(1)

        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_s = x_train[indices].to(device)
        m_s = m_train[indices].to(device)
        y_s = y_train[indices].to(device)

        model.train()
        for epoch in range(epochs):
            loss_accum = 0
            for start in range(0, len(x_s), batch_size):
                end = start + batch_size
                batch_features = x_s[start:end]
                met_features = m_s[start:end]
                lab_features = y_s[start:end]
                
                optimizer.zero_grad()
                out = model(batch_features.float(), met_features.float())
                l = mtlr_neg_log_likelihood(out, lab_features, average=True)
                l.backward()
                optimizer.step()
                loss_accum += l.item()

        output_name = config['training']['output_model_name']
        base, ext = os.path.splitext(output_name)
        save_name = f"{base}_ensemble_seed_{seed}{ext if ext else '.pt'}"
        save_path = os.path.join(output_dir, save_name)
        
        try: 
            torch.save(model.state_dict(), save_path)
            print(f"Model {i+1} saved successfully: {save_name}")
        except Exception as e: 
            print(f"Error saving model state_dict: {e}")

    print("\n--- Ensemble Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config",type=str,default="config.yaml")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    try: main(args)
    except Exception as e:
        print(f"\nAn error occurred during training: {type(e).__name__}: {e}")
        sys.exit(1)