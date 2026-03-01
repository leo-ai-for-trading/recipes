from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class RolloutBuffer:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    log_probs: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    logits: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        *,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        logits: np.ndarray,
    ) -> None:
        self.obs.append(obs.astype(np.float32))
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.logits.append(logits.astype(np.float32))

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.logits.clear()

    def size(self) -> int:
        return len(self.actions)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "obs": np.stack(self.obs).astype(np.float32),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
            "logits": np.stack(self.logits).astype(np.float32),
        }
