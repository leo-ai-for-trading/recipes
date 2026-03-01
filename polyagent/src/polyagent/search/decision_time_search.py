from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from polyagent.belief.belief_state import BeliefState
from polyagent.belief.particle_filter import ParticleBeliefModel
from polyagent.config import EnvConfig, SearchConfig
from polyagent.models.policy_net import PolicyNet
from polyagent.models.value_net import ValueNet


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


@dataclass(slots=True)
class SearchOutput:
    pi_base: np.ndarray
    pi_search: np.ndarray
    q_values: np.ndarray
    kl: float


class DecisionTimeSearcher:
    def __init__(
        self,
        *,
        cfg: SearchConfig,
        env_cfg: EnvConfig,
        belief_model: ParticleBeliefModel,
        value_net: ValueNet,
        device: torch.device,
        seed: int = 7,
    ) -> None:
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.belief_model = belief_model
        self.value_net = value_net
        self.device = device
        self.rng = np.random.default_rng(seed)

    def run(
        self,
        *,
        policy: PolicyNet,
        raw_obs: np.ndarray,
        belief: BeliefState,
    ) -> SearchOutput:
        pi_base = self._policy_probs(policy, raw_obs, belief)
        q_values = self._estimate_q_values(raw_obs=raw_obs, belief=belief)
        pi_search = improve_policy_distribution(
            pi_base=pi_base,
            q_values=q_values,
            alpha_kl=self.cfg.alpha_kl,
            kl_cap=self.cfg.kl_cap,
        )
        return SearchOutput(
            pi_base=pi_base,
            pi_search=pi_search,
            q_values=q_values,
            kl=kl_divergence(pi_search, pi_base),
        )

    def _policy_probs(self, policy: PolicyNet, raw_obs: np.ndarray, belief: BeliefState) -> np.ndarray:
        features = self.belief_model.features(belief)
        obs = np.concatenate([raw_obs.astype(np.float32), features], axis=0)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = policy(obs_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs.astype(np.float32)

    def _estimate_q_values(self, *, raw_obs: np.ndarray, belief: BeliefState) -> np.ndarray:
        n_actions = 6
        particles = self.belief_model.sample(belief, self.cfg.n_rollouts)
        q = np.zeros(n_actions, dtype=np.float32)
        for action in range(n_actions):
            estimates: list[float] = []
            for particle in particles:
                roll = self._rollout_estimate(raw_obs=raw_obs, particle=particle, action=action)
                estimates.append(roll)
            q[action] = float(np.mean(estimates)) if estimates else 0.0
        return q

    def _rollout_estimate(self, *, raw_obs: np.ndarray, particle: np.ndarray, action: int) -> float:
        best_bid = float(raw_obs[0])
        best_ask = float(raw_obs[1])
        mid = float(raw_obs[2])
        spread = float(raw_obs[3])
        inv = float(raw_obs[14])

        drift = float(particle[0]) * 0.002
        vol = abs(float(particle[1])) * 0.002 + 0.0005
        toxicity = float(particle[2]) * 0.001

        inv_delta = self._inventory_delta(action)
        edge = self._execution_edge(action, spread)
        pnl = edge

        sim_mid = mid
        for _ in range(self.cfg.horizon):
            sim_mid += drift + self.rng.normal(0.0, vol) - toxicity * np.sign(inv + inv_delta)
            sim_mid = float(np.clip(sim_mid, 0.0, 1.0))
            pnl += (inv + inv_delta) * (sim_mid - mid)

        # Add value bootstrap using approximated next state.
        next_obs = raw_obs.copy()
        next_obs[0] = max(0.0, sim_mid - spread / 2)
        next_obs[1] = min(1.0, sim_mid + spread / 2)
        next_obs[2] = sim_mid
        next_obs[14] = inv + inv_delta

        features = self.belief_model.features(
            BeliefState(
                particles=np.repeat(particle[None, :], self.belief_model.n_particles, axis=0),
                weights=np.ones(self.belief_model.n_particles, dtype=np.float32) / self.belief_model.n_particles,
            )
        )
        obs_aug = np.concatenate([next_obs.astype(np.float32), features], axis=0)
        obs_t = torch.from_numpy(obs_aug).unsqueeze(0).to(self.device)
        with torch.no_grad():
            boot = float(self.value_net(obs_t).item())
        return pnl + 0.1 * boot

    def _execution_edge(self, action: int, spread: float) -> float:
        half = spread / 2
        qty = self.env_cfg.order_size
        if action == 1:
            return half * qty * 0.7
        if action == 2:
            return -half * qty
        if action == 3:
            return half * qty * 0.7
        if action == 4:
            return -half * qty
        return 0.0

    def _inventory_delta(self, action: int) -> float:
        qty = self.env_cfg.order_size
        if action in (1, 2):
            return qty
        if action in (3, 4):
            return -qty
        return 0.0
