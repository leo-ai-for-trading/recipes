from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from polyagent.config import RLConfig
from polyagent.models.policy_net import PolicyNet
from polyagent.models.value_net import ValueNet


@dataclass(slots=True)
class PPOStats:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    kl: float
    kl_coeff: float
    early_stopped: bool


class PPOKLTrainer:
    def __init__(
        self,
        *,
        policy: PolicyNet,
        value_net: ValueNet,
        cfg: RLConfig,
        device: torch.device,
    ) -> None:
        self.policy = policy
        self.value_net = value_net
        self.cfg = cfg
        self.device = device
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=cfg.lr,
        )
        self.kl_coeff = cfg.kl_coeff_init

    def update(self, batch: dict[str, np.ndarray]) -> PPOStats:
        obs = torch.from_numpy(batch["obs"]).to(self.device)
        actions = torch.from_numpy(batch["actions"]).to(self.device)
        old_log_probs = torch.from_numpy(batch["log_probs"]).to(self.device)
        old_logits = torch.from_numpy(batch["logits"]).to(self.device)
        returns = torch.from_numpy(batch["returns"]).to(self.device)
        advantages = torch.from_numpy(batch["advantages"]).to(self.device)

        n = obs.shape[0]
        idx_all = np.arange(n)

        losses: list[float] = []
        p_losses: list[float] = []
        v_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        early_stopped = False

        for _ in range(self.cfg.epochs):
            np.random.shuffle(idx_all)
            for start in range(0, n, self.cfg.minibatch_size):
                idx = idx_all[start : start + self.cfg.minibatch_size]
                if len(idx) == 0:
                    continue
                mb_obs = obs[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_old_logits = old_logits[idx]
                mb_returns = returns[idx]
                mb_adv = advantages[idx]

                dist = self.policy.dist(mb_obs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                unclipped = ratio * mb_adv
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip, 1.0 + self.cfg.clip) * mb_adv
                policy_loss = -torch.min(unclipped, clipped).mean()

                values = self.value_net(mb_obs)
                value_loss = F.mse_loss(values, mb_returns)

                old_dist = Categorical(logits=mb_old_logits)
                kl = torch.distributions.kl.kl_divergence(old_dist, dist).mean()

                total_loss = (
                    policy_loss
                    + self.cfg.value_coef * value_loss
                    - self.cfg.entropy * entropy
                    + self.kl_coeff * kl
                )

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_net.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.optimizer.step()

                losses.append(float(total_loss.detach().cpu()))
                p_losses.append(float(policy_loss.detach().cpu()))
                v_losses.append(float(value_loss.detach().cpu()))
                entropies.append(float(entropy.detach().cpu()))
                kls.append(float(kl.detach().cpu()))

            epoch_kl = float(np.mean(kls)) if kls else 0.0
            if epoch_kl > 2.0 * self.cfg.kl_target:
                early_stopped = True
                break

        mean_kl = float(np.mean(kls)) if kls else 0.0
        self._adapt_kl_coeff(mean_kl)

        return PPOStats(
            total_loss=float(np.mean(losses)) if losses else 0.0,
            policy_loss=float(np.mean(p_losses)) if p_losses else 0.0,
            value_loss=float(np.mean(v_losses)) if v_losses else 0.0,
            entropy=float(np.mean(entropies)) if entropies else 0.0,
            kl=mean_kl,
            kl_coeff=float(self.kl_coeff),
            early_stopped=early_stopped,
        )

    def action_and_value(self, obs: np.ndarray) -> tuple[int, float, float, np.ndarray]:
        obs_t: Tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.value_net(obs_t)
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
            logits.squeeze(0).cpu().numpy(),
        )

    def value(self, obs: np.ndarray) -> float:
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return float(self.value_net(obs_t).item())

    def _adapt_kl_coeff(self, measured_kl: float) -> None:
        if measured_kl > 1.5 * self.cfg.kl_target:
            self.kl_coeff *= 1.5
        elif measured_kl < self.cfg.kl_target / 1.5:
            self.kl_coeff /= 1.5
        self.kl_coeff = float(np.clip(self.kl_coeff, 1e-4, 1e3))
