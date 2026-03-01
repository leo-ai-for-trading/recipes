from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


def normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (x - x.mean()) / (x.std() + eps)


def advantage_filter_mask(
    advantages: np.ndarray,
    *,
    enabled: bool,
    quantile: float,
    min_abs_adv: float,
) -> np.ndarray:
    n = len(advantages)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if not enabled:
        return np.ones(n, dtype=bool)

    abs_adv = np.abs(advantages)
    q = float(np.quantile(abs_adv, np.clip(quantile, 0.0, 1.0)))
    threshold = max(min_abs_adv, q)
    mask = abs_adv >= threshold
    if not np.any(mask):
        mask[int(np.argmax(abs_adv))] = True
    return mask
