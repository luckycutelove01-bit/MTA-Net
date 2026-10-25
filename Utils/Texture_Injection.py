import torch
import torch.nn as nn
from Parameters import HMS_FEATURE_DIM,PROJECTION_REDUCTION,GAMMA_INITIAL

class HMSProjection(nn.Module):
    def __init__(self, channels: int, hms_dim: int = HMS_FEATURE_DIM,
                 reduction_ratio: int = PROJECTION_REDUCTION, gamma_initial: float = GAMMA_INITIAL):
        super().__init__()
        hidden = max(int(channels) // int(reduction_ratio), 16)
        self.channels = int(channels)
        self.hidden_dim = hidden
        self.norm = nn.LayerNorm(hms_dim)
        self.fc1 = nn.Linear(hms_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_initial), dtype=torch.float32))

    def project(self, hms_features: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.bn1(self.fc1(self.norm(hms_features)))))

    def forward(self, feature: torch.Tensor, hms_features: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = feature.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")
        modulation = self.project(hms_features).view(batch, channels, 1, 1)
        return feature + self.gamma * (feature * modulation)

