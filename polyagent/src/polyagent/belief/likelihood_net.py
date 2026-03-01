from __future__ import annotations

import torch
from torch import nn


class LikelihoodNet(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        dims = [obs_dim + latent_dim, *hidden_dims, 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, particle: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, particle], dim=-1)
        return self.net(x).squeeze(-1)
