# Design Notes: Mapping `2511.07312v1` Ideas to PolyAgent

This document explains how the implementation maps key paper ideas to the repository architecture.

## 1) Imperfect Information + Belief Modeling

Paper concept:
- Markets are partially observed; policy should condition on a belief over hidden state.

Implementation:
- `belief/belief_state.py`: `BeliefState(particles, weights)`
- `belief/particle_filter.py`: particle propagation and Bayesian-style reweighting
- `belief/likelihood_net.py`: small MLP outputs particle-conditioned log-likelihood proxy

In the fast loop:
- Raw observation is augmented with belief summary (particle mean/std) before policy/value inference.

## 2) KL-Regularized Learning (Conservative Updates)

Paper concept:
- Constrain policy updates to avoid destructive policy drift.

Implementation:
- `rl/ppo_kl.py`: PPO clipped objective + explicit KL penalty term
- Adaptive KL coefficient:
  - increase when measured KL > target band
  - decrease when measured KL < target band

This yields conservative, trust-region-like behavior while preserving PPO efficiency.

## 3) Decision-Time Improvement / Search

Paper concept:
- Improve action choice at inference time via belief-conditioned local search.

Implementation:
- `search/decision_time_search.py`
  - start with base policy distribution `pi_base`
  - sample hidden states from belief particles
  - estimate short-horizon action values `Q(a)` with lightweight rollouts + value bootstrap
  - reweight:
    - `pi_search ∝ pi_base * exp(Q / alpha_kl)`
  - enforce KL cap with blending to keep search close to base policy

Used only when enabled (`eval --use-search true` or config).

## 4) Advantage Filtering (High-Signal Updates)

Paper concept:
- Focus updates on informative transitions.

Implementation:
- `rl/utils.py`: `advantage_filter_mask`
- Training path in `cli.py`:
  - computes normalized advantages
  - keeps samples above quantile/min-abs threshold
  - trains PPO on filtered subset

This reduces gradient noise from low-signal transitions.

## 5) Agentic Architecture Split

Execution core (fast path):
- replay env step -> belief update -> policy/value -> optional search -> action

Orchestration layer (slow path):
- `orchestration/llm_agent.py` + `report_templates.py`
- reads run metrics, writes markdown/json reports, proposes next experiments
- never participates in action execution loop

## 6) MVP Boundaries

- Offline replay only (no live exchange routing, no real money).
- Simple deterministic fill logic and lightweight search simulator.
- Designed for quick CPU iteration and reproducible experiments.
