# src/utils.py (Complete file with corrected test block)

import torch
import dask.dataframe as dd
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
import os # For file creation/cleanup in test block
import warnings # To potentially ignore Dask warnings if needed
import shutil # For removing test directory

# Define a type alias for clarity in function signatures (Removed for Python < 3.10 compatibility)
# TensorOrArray: TypeAlias = Union[torch.Tensor, np.ndarray]

# --- Connectivity Matrix Loading ---
def load_connectivity_matrix(
    csv_path: str,
    out_features_s1: int,
    in_features_s1: int,
    c2_input_offset: int = 357692,
    is_one_based: bool = True
    ) -> torch.Tensor:
    """
    Loads the connectivity matrix from a CSV file using Dask, performs processing
    (indexing adjustment, transpose, concatenation with diagonal),
    and validates dimensions.
    (Implementation from previous steps - assuming correct)
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"Connectivity matrix file not found at: {csv_path}")

    print(f"Loading connectivity matrix from: {csv_path} using Dask")
    try:
        # Load using Dask
        ddf = dd.read_csv(csv_path, header=None, delimiter=',', dtype=float, assume_missing=True)
    except Exception as e:
        print(f"Error initializing Dask read_csv for {csv_path}: {e}")
        raise

    # Define processing function for partitions
    def process_partition(df):
        arr = df.values
        if is_one_based:
            valid_mask = ~np.isnan(arr)
            arr[valid_mask] = arr[valid_mask] - 1
        if np.isnan(arr).any():
             print(f"Warning: NaNs detected in connectivity matrix partition. Replacing with -999.")
             arr = np.nan_to_num(arr, nan=-999.0)
        return arr.astype(np.int64)

    # Compute and process
    try:
        print("Processing connectivity matrix partitions...")
        with warnings.catch_warnings(): # Suppress potential Dask warnings if needed
            warnings.simplefilter("ignore")
            conn_mat1_np = ddf.map_partitions(process_partition, meta=np.array([], dtype=np.int64)).compute()
        print(f"Computed connectivity NumPy array shape: {conn_mat1_np.shape}")
    except Exception as e:
        print(f"Error during Dask compute for {csv_path}: {e}")
        raise

    if conn_mat1_np.ndim != 2 or conn_mat1_np.shape[1] != 2:
        raise ValueError(f"Expected 2 columns after loading connectivity CSV, found shape {conn_mat1_np.shape}")

    # Convert to PyTorch tensor
    conn_mat1 = torch.from_numpy(conn_mat1_np).long() # Shape [N, 2] with (Output, Input)

    # Transpose to get shape [2, N]
    conn_mat_loaded = torch.transpose(conn_mat1, 0, 1)
    print(f"Transposed connectivity matrix shape: {conn_mat_loaded.shape}")

    # Create the c2 diagonal connections
    num_c2_connections = out_features_s1
    c2_output_indices = torch.arange(num_c2_connections, dtype=torch.long)
    c2_input_indices = torch.arange(num_c2_connections, dtype=torch.long) + c2_input_offset
    c2 = torch.stack((c2_output_indices, c2_input_indices), dim=0)
    print(f"Generated 'c2' diagonal connections shape: {c2.shape}")

    # Concatenate loaded matrix and c2
    conn_mat_final = torch.cat((conn_mat_loaded, c2), dim=1).long()
    print(f"Final connectivity matrix shape after concatenation: {conn_mat_final.shape}")

    # Validation Checks
    print("Validating final connectivity matrix indices...")
    if conn_mat_final.numel() == 0:
         print("Warning: Final connectivity matrix is empty.")
         return conn_mat_final # Return empty tensor if no connections

    min_out_idx = conn_mat_final[0].min().item()
    max_out_idx = conn_mat_final[0].max().item()
    min_in_idx = conn_mat_final[1].min().item()
    max_in_idx = conn_mat_final[1].max().item()

    print(f"Output index range found: [{min_out_idx}, {max_out_idx}] (Expected: [0, {out_features_s1 - 1}])")
    print(f"Input index range found: [{min_in_idx}, {max_in_idx}] (Expected: [0, {in_features_s1 - 1}])")

    assert min_out_idx >= 0, f"Validation Failed: Min output index {min_out_idx} < 0."
    assert max_out_idx < out_features_s1, f"Validation Failed: Max output index {max_out_idx} >= {out_features_s1}"
    assert min_in_idx >= 0, f"Validation Failed: Min input index {min_in_idx} < 0."
    assert max_in_idx < in_features_s1, f"Validation Failed: Max input index {max_in_idx} >= {in_features_s1}"

    print("Connectivity matrix loaded and validated successfully.")
    return conn_mat_final

# --- Target Encoding ---
def encode_survival(time: Union[torch.Tensor, np.ndarray],
                    event: Union[torch.Tensor, np.ndarray],
                    bins: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Encodes survival time and event status into a target tensor for MTLR loss.
    (Implementation from previous steps - assuming correct)
    """
    # Ensure inputs are torch tensors
    if isinstance(time, np.ndarray): time = torch.from_numpy(np.atleast_1d(time))
    elif isinstance(time, (float, int, np.number)): time = torch.tensor([time], dtype=torch.float32)
    if isinstance(event, np.ndarray): event = torch.from_numpy(np.atleast_1d(event))
    elif isinstance(event, (int, bool, np.bool_, np.number)): event = torch.tensor([event], dtype=torch.int)
    if isinstance(bins, np.ndarray): bins = torch.from_numpy(bins)

    try: device = bins.device
    except AttributeError: device = torch.device("cpu")

    time = time.to(device=device, dtype=torch.float32, non_blocking=True)
    event = event.to(device=device, dtype=torch.int, non_blocking=True)
    bins = bins.to(device=device, dtype=torch.float32, non_blocking=True)

    time = torch.clamp(time, min=0., max=bins.max())
    num_output_bins = bins.shape[0] + 1
    y = torch.zeros((time.shape[0], num_output_bins), dtype=torch.float32, device=device)
    bin_idxs = torch.bucketize(time, bins, right=True)

    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1: y[i, bin_idx] = 1
        else: y[i, bin_idx:] = 1
    return y.squeeze(0) if y.shape[0] == 1 else y


# --- Loss Function Components ---
def masked_logsumexp(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Computes log(sum(exp(x))) over elements identified by mask. Stable.
    (Implementation from previous steps - assuming correct)
    """
    mask_bool = mask.bool()
    masked_x = torch.where(mask_bool, x, torch.full_like(x, -float('inf')))
    max_val, _ = masked_x.max(dim=dim, keepdim=True)
    is_all_inf = torch.isinf(max_val)
    # Replace -inf max_val with 0 for subtraction stability, but handle result later
    max_val_stable = torch.where(is_all_inf, torch.zeros_like(max_val), max_val)
    term = torch.exp(x - max_val_stable)
    sum_exp = torch.sum(term * mask, dim=dim) # Use original mask
    # Clamp sum_exp to avoid log(0)
    log_sum_exp = torch.log(torch.clamp_min(sum_exp, torch.finfo(sum_exp.dtype).tiny)) + max_val_stable.squeeze(dim)
    # Ensure result is -inf if all masked inputs were -inf
    log_sum_exp = torch.where(is_all_inf.squeeze(dim), torch.full_like(log_sum_exp, -float('inf')), log_sum_exp)
    return log_sum_exp

def mtlr_neg_log_likelihood(logits: torch.Tensor, target: torch.Tensor, average: bool = False) -> torch.Tensor:
    """
    Computes the Negative Log Likelihood loss for the MTLR model.
    (Implementation from previous steps - assuming correct)
    """
    target = target.float()
    censored = target.sum(dim=1) > 1; uncensored = ~censored
    device = logits.device # Get device from logits
    nll_censored = masked_logsumexp(logits[censored], target[censored]).sum() if censored.any() else torch.tensor(0., device=device)
    nll_uncensored = (logits[uncensored] * target[uncensored]).sum() if uncensored.any() else torch.tensor(0., device=device)
    norm = torch.logsumexp(logits, dim=1).sum()
    nll_total = -(nll_censored + nll_uncensored - norm)
    if average:
        batch_size = target.size(0)
        if batch_size > 0: nll_total = nll_total / batch_size
        else: return torch.tensor(0., device=device, dtype=logits.dtype)
    return nll_total


# --- Inference Survival Calculation ---
def mtlr_survival(logits: torch.Tensor) -> torch.Tensor:
    """
    Calculates the survival function P(T > t) from MTLR logits.
    (Implementation from previous steps - user provided)
    """
    density = torch.softmax(logits, dim=1)
    G = torch.tril(torch.ones(logits.size(1), logits.size(1), dtype=torch.float32)).to(logits.device)
    survival_probs = torch.matmul(density, G)
    survival_probs = torch.clamp(survival_probs, min=0.0, max=1.0)
    return survival_probs


def load_beta_features(path: str) -> torch.Tensor:
    """Loads beta features from CSV using Dask, returns transposed tensor."""
    if not os.path.exists(path): raise FileNotFoundError(f"Beta features file not found: {path}")
    print(f"Loading beta features from: {path}")
    try:
        ddf = dd.read_csv(path, header=None, delimiter=',', dtype=np.float32, assume_missing=True)
        with warnings.catch_warnings(): # Suppress potential Dask warnings
            warnings.simplefilter("ignore")
            features_np = ddf.compute().values
        features_torch = torch.from_numpy(features_np).float()
        # Original code transposes: rows=samples, columns=features AFTER transpose
        features_torch = torch.transpose(features_torch, 0, 1)
        print(f"Loaded beta features shape: {features_torch.shape}")
        return features_torch
    except Exception as e:
        print(f"Error loading beta features from {path}: {e}")
        raise

def load_cnv_features(path: str) -> torch.Tensor:
    """Loads CNV features from CSV using Dask, returns transposed tensor."""
    if not os.path.exists(path): raise FileNotFoundError(f"CNV features file not found: {path}")
    print(f"Loading CNV features from: {path}")
    try:
        ddf = dd.read_csv(path, header=None, delimiter=',', dtype=np.float32, assume_missing=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features_np = ddf.compute().values
        features_torch = torch.from_numpy(features_np).float()
        # Original code transposes: rows=samples, columns=features AFTER transpose
        features_torch = torch.transpose(features_torch, 0, 1)
        print(f"Loaded CNV features shape: {features_torch.shape}")
        return features_torch
    except Exception as e:
        print(f"Error loading CNV features from {path}: {e}")
        raise

def load_survival_data(path: str, time_col: str, event_col: str, clinical_col: str, id_col: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[np.ndarray]]:
    """Loads survival time, event, and one clinical feature (e.g., metastasis) from CSV using Pandas."""
    if not os.path.exists(path): raise FileNotFoundError(f"Survival/Clinical file not found: {path}")
    print(f"Loading survival/clinical data from: {path}")
    try:
        df = pd.read_csv(path)
        # Convert columns to tensors with appropriate types
        t = torch.tensor(df[time_col].values, dtype=torch.float32)
        e = torch.tensor(df[event_col].values, dtype=torch.int)
        m = torch.tensor(df[clinical_col].values, dtype=torch.float32) # Load as float
        ids = None
        if id_col:
            try:
                ids = df[id_col].values  # Load IDs as numpy array
            except KeyError:
                print(f"Warning: ID column '{id_col}' not found in {path}. Returning None for IDs.")

        print(f"Loaded survival/clinical data for {len(t)} samples." + (
            f" Found IDs for {len(ids)}." if ids is not None else " No IDs requested/found."))
        return t, e, m, ids  # Return IDs
    except KeyError as err:
        print(f"Error: Column '{err}' not found in {path}")
        raise
    except Exception as err:
        print(f"Error reading survival/clinical CSV {path}: {err}")
        raise

def normalize_instance_wise(data: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Applies instance-wise normalization (z-score per sample).
    Calculates mean and std dev for each sample across its features independently.

    Args:
        data (torch.Tensor): Data to normalize (shape: [n_samples, n_features]).
        eps (float): Small value added to std dev for stability.

    Returns:
        torch.Tensor: Normalized data.
    """
    if data.ndim != 2:
        raise ValueError(f"Input data must be 2D (samples, features). Got shape: {data.shape}")
    if data.numel() == 0:
         print("Warning: Input data for normalization is empty.")
         return data # Return empty tensor

    print(f"Applying instance-wise normalization to data of shape: {data.shape}")
    # Calculate mean across feature dimension (dim=1) for each sample
    sample_mean = torch.mean(data, dim=1, keepdim=True) # Shape [n_samples, 1]

    # Subtract the sample's own mean
    data_centered = data - sample_mean # Broadcasting happens

    # Calculate std dev across feature dimension (dim=1) for each sample
    # Using unbiased=False to match torch.std default when dim is specified
    sample_std = torch.std(data_centered, dim=1, keepdim=True, unbiased=False) # Shape [n_samples, 1]

    # Handle samples with zero standard deviation (e.g., constant features)
    # Avoid division by zero by using the epsilon, or potentially clamping std dev
    sample_std = torch.clamp_min(sample_std, eps) # Clamp std dev instead of just adding eps? Adding eps is common.

    # Normalize by the sample's own std dev
    data_norm = data_centered / (sample_std + eps) # Add epsilon for stability

    # Check for NaNs/Infs which might occur if std dev was effectively zero even with eps
    if torch.isnan(data_norm).any() or torch.isinf(data_norm).any():
        num_nan = torch.isnan(data_norm).sum().item()
        num_inf = torch.isinf(data_norm).sum().item()
        print(f"Warning: NaNs ({num_nan}) or Infs ({num_inf}) detected after instance-wise normalization. Check input data or epsilon.")
        # Optionally replace NaNs/Infs, e.g., with 0
        # data_norm = torch.nan_to_num(data_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return data_norm


# --- Test block ---
if __name__ == '__main__':
    print("\n--- Running src/utils.py tests ---")
    # Use a temporary directory for dummy files
    test_dir = "temp_utils_test_data"
    if os.path.exists(test_dir): shutil.rmtree(test_dir) # Clean up previous run if needed
    os.makedirs(test_dir)
    print(f"Created temporary test directory: {test_dir}")

    # --- Test load_connectivity_matrix ---
    print("\nTesting load_connectivity_matrix...")
    dummy_csv_path_conn = os.path.join(test_dir, "dummy_conn_mat_utils_test.csv")
    dummy_in_features_conn = 500
    dummy_out_features_conn = 50
    try:
        # Create dummy CSV with (Output, Input), 1-based
        dummy_conns_conn = np.stack([
            np.random.randint(1, dummy_out_features_conn + 1, 1000), # 1 to 50
            np.random.randint(1, dummy_in_features_conn + 1, 1000)  # 1 to 500
        ], axis=1)
        pd.DataFrame(dummy_conns_conn).to_csv(dummy_csv_path_conn, header=False, index=False)
        # print(f"Created dummy CSV: {dummy_csv_path_conn}")

        conn_mat = load_connectivity_matrix(
            csv_path=dummy_csv_path_conn,
            out_features_s1=dummy_out_features_conn,
            in_features_s1=dummy_in_features_conn,
            c2_input_offset=300,
            is_one_based=True
        )
        print(f"Successfully loaded dummy conn_mat, shape: {conn_mat.shape}")
        assert conn_mat.shape == (2, 1000 + dummy_out_features_conn)
        assert conn_mat.dtype == torch.long
        # Add check for 0-based indexing after load
        assert conn_mat.min() >= 0
        print("load_connectivity_matrix test passed.")
    except Exception as e:
        print(f"Error testing load_connectivity_matrix: {e}")
        # raise # Comment out raise during debugging if needed
    finally:
        # Cleanup handled by shutil.rmtree at the end
        pass


    # --- Test Low-Level Loaders ---
    print("\nTesting low-level data loaders...")
    dummy_beta_path = os.path.join(test_dir, "dummy_beta.csv")
    dummy_cnv_path = os.path.join(test_dir, "dummy_cnv.csv")
    dummy_surv_path = os.path.join(test_dir, "dummy_surv.csv")
    n_samples_ll = 5
    n_beta_feat = 10
    n_cnv_feat = 5
    try:
        # Create dummy beta CSV (features x samples before transpose)
        pd.DataFrame(np.random.rand(n_beta_feat, n_samples_ll).astype(np.float32)).to_csv(dummy_beta_path, header=False, index=False)
        # Create dummy cnv CSV (features x samples before transpose)
        pd.DataFrame(np.random.rand(n_cnv_feat, n_samples_ll).astype(np.float32)).to_csv(dummy_cnv_path, header=False, index=False)
        # Create dummy survival CSV
        pd.DataFrame({
            'TIME': np.random.rand(n_samples_ll) * 10,
            'EVENT': np.random.randint(0, 2, n_samples_ll),
            'METASTASIS': np.random.randint(0, 2, n_samples_ll).astype(np.float32)
        }).to_csv(dummy_surv_path, index=False)
        # print(f"Created dummy data files in {test_dir}")

        # Test loaders
        beta_data = load_beta_features(dummy_beta_path)
        assert beta_data.shape == (n_samples_ll, n_beta_feat)
        assert beta_data.dtype == torch.float32
        print("load_beta_features test passed.")

        cnv_data = load_cnv_features(dummy_cnv_path)
        assert cnv_data.shape == (n_samples_ll, n_cnv_feat)
        assert cnv_data.dtype == torch.float32
        print("load_cnv_features test passed.")

        t_surv, e_surv, m_surv = load_survival_data(dummy_surv_path, 'TIME', 'EVENT', 'METASTASIS')
        assert t_surv.shape == (n_samples_ll,) and t_surv.dtype == torch.float32
        assert e_surv.shape == (n_samples_ll,) and e_surv.dtype == torch.int
        assert m_surv.shape == (n_samples_ll,) and m_surv.dtype == torch.float32
        print("load_survival_data test passed.")

    except Exception as e:
        print(f"Error testing low-level loaders: {e}")
        # raise

    # --- Test normalize_features ---
    print("\nTesting normalize_instance_wise...")
    try:
        n_samples_norm_test = 5
        n_features_norm_test = 10
        # Create dummy data with known means/std devs if possible, or just random
        dummy_data_norm = torch.randn(n_samples_norm_test, n_features_norm_test, dtype=torch.float32)
        # Add an offset to make means non-zero
        dummy_data_norm += torch.randn(n_samples_norm_test, 1) * 5
        # Make one sample constant to test stability
        dummy_data_norm[1, :] = 3.0

        normalized_data = normalize_instance_wise(dummy_data_norm)

        assert normalized_data.shape == dummy_data_norm.shape
        # Check if mean is close to 0 and std dev is close to 1 per instance
        # Need to handle the constant row (std dev = 0, mean = 3) - result should be ~0 after normalization
        assert torch.allclose(torch.mean(normalized_data[0], dim=0), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(torch.std(normalized_data[0], dim=0, unbiased=False), torch.tensor(1.0), atol=1e-6)
        # Check constant row - mean should be original mean, std should be 0. Normalization should result in 0s.
        assert torch.allclose(normalized_data[1], torch.zeros(n_features_norm_test), atol=1e-6)
        # Check another random row
        assert torch.allclose(torch.mean(normalized_data[2], dim=0), torch.tensor(0.0), atol=1e-6)
        assert torch.allclose(torch.std(normalized_data[2], dim=0, unbiased=False), torch.tensor(1.0), atol=1e-6)

        print("normalize_instance_wise test passed.")
    except Exception as e:
        print(f"Error testing normalize_instance_wise: {e}")

    # --- Test encode_survival ---
    print("\nTesting encode_survival...")
    test_times_enc = np.array([1.2, 5.5, 0.1, 10.0, 7.0])
    test_events_enc = np.array([1, 0, 1, 0, 1])
    test_bins_enc = np.array([1.0, 2.0, 5.0, 8.0]) # 4 bins -> 5 output columns
    expected_target_enc = torch.tensor([
        [0., 1., 0., 0., 0.], [0., 0., 0., 1., 1.], [1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1.], [0., 0., 0., 1., 0.] ], dtype=torch.float32)
    try:
        encoded_target = encode_survival(test_times_enc, test_events_enc, test_bins_enc)
        assert encoded_target.shape == (len(test_times_enc), len(test_bins_enc) + 1)
        assert torch.allclose(encoded_target, expected_target_enc)
        print("encode_survival test passed.")
    except Exception as e: print(f"Error testing encode_survival: {e}"); # raise

    # --- Test Loss and Survival Functions ---
    print("\nTesting mtlr_neg_log_likelihood and mtlr_survival...")
    batch_size_loss = 4
    num_time_bins_loss = 15
    num_output_dims_loss = num_time_bins_loss + 1 # K = 16
    dummy_logits_loss = torch.randn(batch_size_loss, num_output_dims_loss) # Shape [4, 16]

    # --- CORRECTED Dummy target tensor generation ---
    print(f"Generating dummy target for loss test with shape: ({batch_size_loss}, {num_output_dims_loss})")
    dummy_target_loss = torch.zeros(batch_size_loss, num_output_dims_loss, dtype=torch.float32)
    # Example data points (ensure indices are within 0 to 15, matching K=16 output dims)
    if num_output_dims_loss > 5: dummy_target_loss[0, 5] = 1.0 # Event at bin 5
    if num_output_dims_loss > 11: dummy_target_loss[1, 11:] = 1.0 # Censored after bin 10
    if num_output_dims_loss > 0: dummy_target_loss[2, 0] = 1.0 # Event at bin 0
    if num_output_dims_loss > 2: dummy_target_loss[3, 2:] = 1.0 # Censored after bin 1
    print("Dummy target for loss test generated.")
    # --- End Correction ---

    try:
        # Test loss calculation (using the CORRECTED dummy_target_loss)
        loss = mtlr_neg_log_likelihood(dummy_logits_loss, dummy_target_loss, average=False)
        loss_avg = mtlr_neg_log_likelihood(dummy_logits_loss, dummy_target_loss, average=True)
        print(f"Loss calculation successful. Total: {loss.item()}, Average: {loss_avg.item()}")
        assert torch.is_tensor(loss) and loss.numel() == 1
        # Use isclose for float comparison
        assert torch.isclose(loss_avg, loss / batch_size_loss if batch_size_loss > 0 else torch.tensor(0.))

        # Test survival calculation
        survival_output = mtlr_survival(dummy_logits_loss)
        print(f"mtlr_survival output shape: {survival_output.shape}")
        expected_survival_shape = (batch_size_loss, num_output_dims_loss)
        assert survival_output.shape == expected_survival_shape
        assert survival_output.min() >= 0.0 and survival_output.max() <= 1.0 + 1e-6
        assert torch.all(survival_output[:, :-1] >= survival_output[:, 1:] - 1e-6)
        print("mtlr_neg_log_likelihood and mtlr_survival tests passed.")

    except Exception as e:
        print(f"Error testing loss/survival functions: {e}")
        # raise

    # Clean up test directory
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
        print(f"Removed temporary test directory: {test_dir}")

    print("\n--- src/utils.py tests completed ---")