from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from polyagent.models.encoders import ObsEncoder


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = ObsEncoder(obs_dim=obs_dim, hidden_dim=hidden_dim)
        self.logits_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        return self.logits_head(z)

    def dist(self, obs: torch.Tensor) -> Categorical:
        logits = self.forward(obs)
        return Categorical(logits=logits)
