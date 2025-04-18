# src/model.py

import torch
import torch.nn as nn
import sparselinear as sl  # Assuming sparselinear is installed
from typing import List, Optional, Tuple

# --- MTLR (Multi-task Logistic Regression for Survival Analysis) ---
class MTLR(nn.Module):
    """
    Multi-task Logistic Regression for Survival Analysis layer.
    Predicts parameters (logits) for a discrete-time survival model.
    """
    def __init__(self, in_features: int, num_time_bins: int):
        """
        Initializes the MTLR layer.

        Args:
            in_features (int): Number of input features from the previous layer.
            num_time_bins (int): The number of desired discrete time intervals (K-1).
                                 The model outputs K = num_time_bins + 1 logits.
        """
        super().__init__()

        self.in_features = in_features
        self.num_time_points = num_time_bins + 1 # K points define K-1 intervals
        self.num_intervals = num_time_bins      # K-1 intervals

        # Parameters for the linear transformation (predicting K-1 internal scores)
        self.mtlr_weight = nn.Parameter(torch.Tensor(self.in_features, self.num_intervals))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_intervals))

        # Transformation matrix G to get K logits from K-1 scores
        # Shape [num_intervals, num_time_points] = [K-1, K]
        G = torch.tril(torch.ones(self.num_intervals, self.num_intervals, requires_grad=False))
        G_extended = torch.cat((G, torch.zeros(self.num_intervals, 1)), dim=1)
        self.register_buffer("G", G_extended) # Shape [K-1, K]

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MTLR layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features].

        Returns:
            torch.Tensor: Output tensor representing raw logits for each time interval.
                          Shape [batch_size, num_time_points] ([batch_size, K = num_time_bins + 1]).
                          These logits are the direct input for the mtlr_neg_log_likelihood loss
                          and the mtlr_survival inference function.
        """
        # Calculate the K-1 internal scores
        internal_scores = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias # Shape: [batch_size, K-1]

        # Apply the transformation matrix G to get K logits
        # Output shape: [batch_size, K-1] @ [K-1, K] -> [batch_size, K]
        logits = torch.matmul(internal_scores, self.G)

        # Return raw logits directly (no prepended zero)
        return logits # Shape: [batch_size, K] = [batch_size, num_time_bins + 1]

    def reset_parameters(self):
        """Initializes the layer's weights and biases."""
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        """String representation of the layer."""
        return (f"{self.__class__.__name__}(in_features={self.in_features},"
                f" num_time_bins={self.num_intervals})") # Report intervals (K-1)


# --- Part 1: Feature Extractor ---
class Part1_FeatureExtractor(nn.Module):
    """
    First part of the combined model.
    Processes high-dimensional sparse input using a SparseLinear layer
    followed by several dense layers with ELU activations and LayerNorm.
    """
    def __init__(self,
                 input_dim: int,
                 conn_mat: torch.Tensor,
                 layer_dims: List[int],
                 dropout_rate: float):
        """
        Initializes Part 1 of the model.

        Args:
            input_dim (int): Dimensionality of the input features.
            conn_mat (torch.Tensor): Connectivity matrix for the SparseLinear layer.
                                    Should be a LongTensor of shape [2, num_connections].
            layer_dims (List[int]): List containing the output dimensions of each layer
                                    starting from the SparseLinear layer.
                                    Example: [16067, 1602, 421, 64]
            dropout_rate (float): Dropout probability for the dropout layer after the
                                  first SparseLinear layer.
        """
        super().__init__()

        if not isinstance(layer_dims, list) or len(layer_dims) != 4:
            raise ValueError("layer_dims must be a list of 4 integers.")

        self.layer_dims = layer_dims

        # Layer 1: Sparse Input Layer
        self.s1 = sl.SparseLinear(input_dim, layer_dims[0], connectivity=conn_mat)
        self.e1 = nn.ELU()
        self.l1 = nn.LayerNorm(layer_dims[0])
        self.dx = nn.Dropout(dropout_rate) # Dropout after first layer block

        # Layer 2
        self.s2 = nn.Linear(layer_dims[0], layer_dims[1])
        self.e2 = nn.ELU()
        self.l2 = nn.LayerNorm(layer_dims[1])

        # Layer 3
        self.s3 = nn.Linear(layer_dims[1], layer_dims[2])
        self.e3 = nn.ELU()
        self.l3 = nn.LayerNorm(layer_dims[2])

        # Layer 4: Output Layer of Part 1
        self.s4 = nn.Linear(layer_dims[2], layer_dims[3])
        self.e4 = nn.ELU()
        self.l4 = nn.LayerNorm(layer_dims[3]) # Final LayerNorm before output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Part 1.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, layer_dims[3]].
        """
        # Layer 1 Block
        x = self.s1(x)
        x = self.e1(x)
        x = self.l1(x)
        x = self.dx(x) # Apply dropout

        # Layer 2 Block
        x = self.s2(x)
        x = self.e2(x)
        x = self.l2(x)

        # Layer 3 Block
        x = self.s3(x)
        x = self.e3(x)
        x = self.l3(x)

        # Layer 4 Block
        x = self.s4(x)
        x = self.e4(x)
        x = self.l4(x)

        return x


# --- Part 2: Survival Prediction Head ---
class Part2_SurvivalHead(nn.Module):
    """
    Second part of the combined model (Survival Prediction Head).
    Takes the concatenated features and predicts survival logits using MTLR.
    Outputs raw logits suitable for mtlr_neg_log_likelihood loss.
    """
    def __init__(self,
                 input_dim: int,
                 num_time_bins: int,
                 dropout_rate: float):
        """
        Initializes Part 2 of the model.

        Args:
            input_dim (int): Dimensionality of the input features coming from
                             Part 1 + concatenated clinical features.
                             (e.g., 64 + 1 = 65).
            num_time_bins (int): The number of discrete time bins (K-1 intervals).
            dropout_rate (float): Dropout probability applied before the MTLR layer.
        """
        super().__init__()
        self.d1 = nn.Dropout(dropout_rate)
        # Use the locally defined MTLR class
        self.m1 = MTLR(in_features=input_dim, num_time_bins=num_time_bins)
        # REMOVED self.log_softmax

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Part 2.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].

        Returns:
            torch.Tensor: Output tensor of raw logits for survival across time bins.
                          Shape [batch_size, num_time_bins + 1].
        """
        x = self.d1(x)
        x = self.m1(x) # Output shape: [batch_size, num_time_bins + 1]
        # REMOVED log_softmax call
        return x


# --- Combined Model ---
class CombinedSurvivalModel(nn.Module):
    # ... (docstring) ...
    def __init__(self,
                 part1_input_dim: int,
                 conn_mat: torch.Tensor,
                 part1_layer_dims: List[int],
                 part1_dropout_rate: float,
                 num_clinical_features: int,
                 clinical_feature_weight: float,
                 part2_num_time_bins: int,
                 part2_dropout_rate: float):
        # ... (docstring) ...
        super().__init__()

        # --- RENAME these attributes back to p1 and p2 ---
        # Instantiate Part 1
        self.p1 = Part1_FeatureExtractor( # Rename self.part1 -> self.p1
            input_dim=part1_input_dim,
            conn_mat=conn_mat,
            layer_dims=part1_layer_dims,
            dropout_rate=part1_dropout_rate
        )

        # Store weight for clinical feature(s) - KEEP THIS
        self.register_buffer('clinical_weight', torch.tensor(clinical_feature_weight))

        # Calculate input dimension for Part 2
        part2_input_dim = part1_layer_dims[-1] + num_clinical_features

        # Instantiate Part 2
        self.p2 = Part2_SurvivalHead( # Rename self.part2 -> self.p2
            input_dim=part2_input_dim,
            num_time_bins=part2_num_time_bins,
            dropout_rate=part2_dropout_rate
        )
        # --- End Renaming ---

    def forward(self, x_main: torch.Tensor, x_clinical: torch.Tensor) -> torch.Tensor:
        # --- UPDATE forward method to use self.p1 and self.p2 ---
        # Process main features through Part 1
        features = self.p1(x_main) # Use self.p1

        # Apply weight to clinical features
        # ... (weighting logic remains the same, using self.clinical_weight) ...
        if x_clinical.ndim == 1: x_clinical = x_clinical.unsqueeze(1)
        if x_clinical.shape[1] != self.clinical_weight.numel() and x_clinical.shape[1] != 1 :
             print(f"Warning: Mismatch num_clinical_features/weight size")
        weighted_clinical = x_clinical * self.clinical_weight

        # Concatenate features
        combined_features = torch.cat((features, weighted_clinical), dim=1)

        # Process combined features through Part 2 (Survival Head)
        output_logits = self.p2(combined_features) # Use self.p2

        return output_logits

# --- Example Usage or Test ---
if __name__ == '__main__':
    # This block executes only when the script is run directly (e.g., python src/model.py)
    # It's useful for basic testing of the model definitions.

    print("Testing model definitions (outputting raw logits)...")

    # --- Dummy Data and Parameters ---
    batch_size = 4
    p1_input_dim = 373759 # Number of input features for s1
    p1_out_features_s1 = 16067 # Number of output features for s1
    # Using smaller num_connections for faster dummy data generation in test
    num_connections = 100000 # Reduced from original for test speed
    num_clinical_feats = 1

    # --- Dummy Connectivity Matrix Generation ---
    print(f"Generating dummy connectivity matrix with {num_connections} connections...")
    # Row indices (output features): range [0, p1_out_features_s1 - 1]
    dummy_row_indices = torch.randint(0, p1_out_features_s1, (num_connections,), dtype=torch.long)
    # Col indices (input features): range [0, p1_input_dim - 1]
    dummy_col_indices = torch.randint(0, p1_input_dim, (num_connections,), dtype=torch.long)
    # Combine into the correct shape [2, num_connections]
    dummy_conn_mat = torch.stack((dummy_row_indices, dummy_col_indices), dim=0)
    print(f"Dummy connectivity matrix shape: {dummy_conn_mat.shape}")
    print(f"Max index in row 0 (output features): {dummy_conn_mat[0].max().item()} (should be < {p1_out_features_s1})")
    print(f"Max index in row 1 (input features): {dummy_conn_mat[1].max().item()} (should be < {p1_input_dim})")

    # --- Model Hyperparameters ---
    p1_layer_dims = [p1_out_features_s1, 1602, 421, 64] # Use the variable
    p1_dropout = 0.2723
    clinical_w = -4.4
    p2_time_bins = 15 # K-1 = 15 intervals -> K = 16 output logits
    p2_dropout = 0.2927

    # --- Instantiate Model ---
    print("Instantiating CombinedSurvivalModel...")
    try:
        model = CombinedSurvivalModel(
            part1_input_dim=p1_input_dim,
            conn_mat=dummy_conn_mat, # Use the correctly generated dummy matrix
            part1_layer_dims=p1_layer_dims,
            part1_dropout_rate=p1_dropout,
            num_clinical_features=num_clinical_feats,
            clinical_feature_weight=clinical_w,
            part2_num_time_bins=p2_time_bins,
            part2_dropout_rate=p2_dropout
        )
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Error during model instantiation: {e}")
        raise

    # --- Test Forward Pass ---
    print("Testing forward pass...")
    # Using smaller dummy input features matching test conn_mat size is faster
    # But for correctness check use original dimensions
    dummy_x_main = torch.randn(batch_size, p1_input_dim)
    dummy_x_clinical = torch.randn(batch_size, num_clinical_feats)

    try:
        model.eval() # Set model to evaluation mode
        with torch.no_grad(): # Disable gradient calculation
            output_logits = model(dummy_x_main, dummy_x_clinical) # Output is now raw logits

        # Output shape should be [batch_size, num_time_bins + 1] (K = 16)
        expected_output_shape = (batch_size, p2_time_bins + 1)
        print(f"Forward pass successful. Raw Logits Output shape: {output_logits.shape}. Expected: {expected_output_shape}")
        assert output_logits.shape == expected_output_shape, "Output shape mismatch!"

    except Exception as e:
        print(f"Error during forward pass: {e}")
        raise

    print("Model definition tests completed successfully.")