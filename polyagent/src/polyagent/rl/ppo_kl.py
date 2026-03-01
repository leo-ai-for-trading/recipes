from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    kl: float


class PPOKLTrainer:
    def __init__(self) -> None:
        self.ready = True

    def train_step(self) -> PPOStats:
        _ = torch.tensor(0.0)
        return PPOStats(policy_loss=0.0, value_loss=0.0, entropy=0.0, kl=0.0)
