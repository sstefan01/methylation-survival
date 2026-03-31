import torch
import torch.nn as nn
import sparselinear as sl
from typing import List


class MTLR(nn.Module):
    """Multi-task Logistic Regression survival layer."""

    def __init__(self, in_features: int, num_time_bins: int):
        super().__init__()
        self.in_features = in_features
        self.num_time_points = num_time_bins + 1
        self.num_intervals = num_time_bins
        self.mtlr_weight = nn.Parameter(torch.Tensor(self.in_features, self.num_intervals))
        self.mtlr_bias = nn.Parameter(torch.Tensor(self.num_intervals))
        g = torch.tril(torch.ones(self.num_intervals, self.num_intervals, requires_grad=False))
        g_extended = torch.cat((g, torch.zeros(self.num_intervals, 1)), dim=1)
        self.register_buffer("G", g_extended)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        internal_scores = torch.matmul(x, self.mtlr_weight) + self.mtlr_bias
        return torch.matmul(internal_scores, self.G)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.mtlr_weight)
        nn.init.constant_(self.mtlr_bias, 0.)

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, num_time_bins={self.num_intervals})"


class Part1_FeatureExtractor(nn.Module):
    """Sparse + dense feature extractor shared by all survival heads."""

    def __init__(self, input_dim: int, conn_mat: torch.Tensor, layer_dims: List[int], dropout_rate: float):
        super().__init__()
        if not isinstance(layer_dims, list) or len(layer_dims) != 4:
            raise ValueError("layer_dims must be a list of 4 integers.")
        self.layer_dims = layer_dims
        self.s1 = sl.SparseLinear(input_dim, layer_dims[0], connectivity=conn_mat)
        self.e1 = nn.ELU()
        self.l1 = nn.LayerNorm(layer_dims[0])
        self.dx = nn.Dropout(dropout_rate)
        self.s2 = nn.Linear(layer_dims[0], layer_dims[1])
        self.e2 = nn.ELU()
        self.l2 = nn.LayerNorm(layer_dims[1])
        self.s3 = nn.Linear(layer_dims[1], layer_dims[2])
        self.e3 = nn.ELU()
        self.l3 = nn.LayerNorm(layer_dims[2])
        self.s4 = nn.Linear(layer_dims[2], layer_dims[3])
        self.e4 = nn.ELU()
        self.l4 = nn.LayerNorm(layer_dims[3])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dx(self.l1(self.e1(self.s1(x))))
        x = self.l2(self.e2(self.s2(x)))
        x = self.l3(self.e3(self.s3(x)))
        x = self.l4(self.e4(self.s4(x)))
        return x


class Part2_MTLRHead(nn.Module):
    """MTLR head that outputs discrete-time survival logits."""

    def __init__(self, input_dim: int, num_time_bins: int, dropout_rate: float):
        super().__init__()
        self.d1 = nn.Dropout(dropout_rate)
        self.m1 = MTLR(in_features=input_dim, num_time_bins=num_time_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m1(self.d1(x))


class Part2_DeepSurvHead(nn.Module):
    """DeepSurv head that outputs a single scalar risk score per sample."""

    def __init__(self, input_dim: int, dropout_rate: float):
        super().__init__()
        self.d1 = nn.Dropout(dropout_rate)
        self.out = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.d1(x))


class CombinedSurvivalModel(nn.Module):
    """Combined sparse feature extractor plus configurable survival head."""

    SUPPORTED_HEAD_TYPES = {"mtlr", "deepsurv"}

    def __init__(
        self,
        part1_input_dim: int,
        conn_mat: torch.Tensor,
        part1_layer_dims: List[int],
        part1_dropout_rate: float,
        num_clinical_features: int,
        clinical_feature_weight: float,
        part2_num_time_bins: int,
        part2_dropout_rate: float,
        survival_head_type: str = "mtlr",
    ):
        super().__init__()
        self.survival_head_type = survival_head_type.lower()
        if self.survival_head_type not in self.SUPPORTED_HEAD_TYPES:
            raise ValueError(
                f"Unsupported survival_head_type '{survival_head_type}'. Supported values: {sorted(self.SUPPORTED_HEAD_TYPES)}"
            )

        self.p1 = Part1_FeatureExtractor(
            input_dim=part1_input_dim,
            conn_mat=conn_mat,
            layer_dims=part1_layer_dims,
            dropout_rate=part1_dropout_rate,
        )
        self.register_buffer("clinical_weight", torch.tensor(clinical_feature_weight))
        part2_input_dim = part1_layer_dims[-1] + num_clinical_features

        if self.survival_head_type == "mtlr":
            self.p2 = Part2_MTLRHead(
                input_dim=part2_input_dim,
                num_time_bins=part2_num_time_bins,
                dropout_rate=part2_dropout_rate,
            )
        else:
            self.p2 = Part2_DeepSurvHead(
                input_dim=part2_input_dim,
                dropout_rate=part2_dropout_rate,
            )

    def forward(self, x_main: torch.Tensor, x_clinical: torch.Tensor) -> torch.Tensor:
        features = self.p1(x_main)
        if x_clinical.ndim == 1:
            x_clinical = x_clinical.unsqueeze(1)
        if x_clinical.shape[1] != self.clinical_weight.numel() and x_clinical.shape[1] != 1:
            print("Warning: Mismatch num_clinical_features/weight size")
        weighted_clinical = x_clinical * self.clinical_weight
        combined_features = torch.cat((features, weighted_clinical), dim=1)
        return self.p2(combined_features)


if __name__ == "__main__":
    print("Testing model definitions...")
    batch_size = 4
    p1_input_dim = 373759
    p1_out_features_s1 = 16067
    num_connections = 100000
    num_clinical_feats = 1
    dummy_row_indices = torch.randint(0, p1_out_features_s1, (num_connections,), dtype=torch.long)
    dummy_col_indices = torch.randint(0, p1_input_dim, (num_connections,), dtype=torch.long)
    dummy_conn_mat = torch.stack((dummy_row_indices, dummy_col_indices), dim=0)
    p1_layer_dims = [p1_out_features_s1, 1602, 421, 64]
    p1_dropout = 0.2723
    clinical_w = -4.4
    p2_time_bins = 15
    p2_dropout = 0.2927
    dummy_x_main = torch.randn(batch_size, p1_input_dim)
    dummy_x_clinical = torch.randn(batch_size, num_clinical_feats)

    for head_type, expected_shape in (("mtlr", (batch_size, p2_time_bins + 1)), ("deepsurv", (batch_size, 1))):
        print(f"Instantiating CombinedSurvivalModel with head='{head_type}'...")
        model = CombinedSurvivalModel(
            part1_input_dim=p1_input_dim,
            conn_mat=dummy_conn_mat,
            part1_layer_dims=p1_layer_dims,
            part1_dropout_rate=p1_dropout,
            num_clinical_features=num_clinical_feats,
            clinical_feature_weight=clinical_w,
            part2_num_time_bins=p2_time_bins,
            part2_dropout_rate=p2_dropout,
            survival_head_type=head_type,
        )
        model.eval()
        with torch.no_grad():
            output = model(dummy_x_main, dummy_x_clinical)
        print(f"Output shape for {head_type}: {output.shape}")
        assert output.shape == expected_shape, f"Output shape mismatch for {head_type}!"

    print("Model definition tests completed successfully.")
