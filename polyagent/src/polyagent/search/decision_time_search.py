from __future__ import annotations

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def improve_policy_distribution(
    *,
    pi_base: np.ndarray,
    q_values: np.ndarray,
    alpha_kl: float,
    kl_cap: float,
) -> np.ndarray:
    logits = np.log(np.clip(pi_base, 1e-12, 1.0)) + q_values / max(alpha_kl, 1e-6)
    logits -= np.max(logits)
    pi_search = np.exp(logits)
    pi_search /= np.sum(pi_search)

    kl = kl_divergence(pi_search, pi_base)
    if kl <= kl_cap:
        return pi_search

    blend = min(1.0, kl_cap / max(kl, 1e-8))
    mixed = blend * pi_search + (1.0 - blend) * pi_base
    mixed /= np.sum(mixed)
    return mixed
