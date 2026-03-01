from __future__ import annotations

import numpy as np
import torch

from polyagent.belief.belief_state import BeliefState
from polyagent.belief.likelihood_net import LikelihoodNet


class ParticleBeliefModel:
    def __init__(
        self,
        *,
        n_particles: int,
        latent_dim: int,
        transition_noise: float,
        likelihood_net: LikelihoodNet,
        seed: int = 7,
    ) -> None:
        self.n_particles = n_particles
        self.latent_dim = latent_dim
        self.transition_noise = transition_noise
        self.likelihood_net = likelihood_net
        self.rng = np.random.default_rng(seed)

    def init_belief(self) -> BeliefState:
        particles = self.rng.normal(0.0, 1.0, size=(self.n_particles, self.latent_dim)).astype(np.float32)
        weights = np.ones(self.n_particles, dtype=np.float32) / self.n_particles
        return BeliefState(particles=particles, weights=weights)

    @torch.no_grad()
    def update(self, belief: BeliefState, obs: np.ndarray) -> BeliefState:
        particles = belief.particles + self.rng.normal(
            0.0,
            self.transition_noise,
            size=belief.particles.shape,
        ).astype(np.float32)
        obs_batch = np.repeat(obs[None, :], self.n_particles, axis=0).astype(np.float32)

        obs_t = torch.from_numpy(obs_batch)
        part_t = torch.from_numpy(particles)
        log_like = self.likelihood_net(obs_t, part_t).cpu().numpy()
        log_w = np.log(np.clip(belief.weights, 1e-12, 1.0)) + log_like
        log_w -= np.max(log_w)
        w = np.exp(log_w)
        w /= np.sum(w)
        return BeliefState(particles=particles, weights=w.astype(np.float32))

    def sample(self, belief: BeliefState, n: int) -> np.ndarray:
        idx = self.rng.choice(len(belief.particles), size=n, p=belief.weights, replace=True)
        return belief.particles[idx]

    def features(self, belief: BeliefState) -> np.ndarray:
        mean = np.sum(belief.particles * belief.weights[:, None], axis=0)
        centered = belief.particles - mean[None, :]
        var = np.sum((centered**2) * belief.weights[:, None], axis=0)
        std = np.sqrt(np.clip(var, 1e-8, None))
        return np.concatenate([mean, std]).astype(np.float32)
