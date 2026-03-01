from __future__ import annotations

import torch
from torch import nn

from polyagent.models.encoders import ObsEncoder


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = ObsEncoder(obs_dim=obs_dim, hidden_dim=hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        return self.value_head(z).squeeze(-1)
