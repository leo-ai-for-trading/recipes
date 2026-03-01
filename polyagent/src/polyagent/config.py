from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvConfig(BaseModel):
    tick_size: float = 0.01
    book_levels: int = 3
    maker_fee_bps: float = 1.0
    taker_fee_bps: float = 2.0
    inventory_penalty: float = 0.0005
    impact_penalty: float = 0.0002
    step_interval_ms: int = 1000
    episode_length: int = 256
    order_size: float = 10.0


class ModelConfig(BaseModel):
    hidden_dim: int = 128
    obs_dim: int = 20
    action_dim: int = 6


class RLConfig(BaseModel):
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    kl_target: float = 0.02
    kl_coeff_init: float = 0.5
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    batch_size: int = 256
    minibatch_size: int = 64
    epochs: int = 4
    train_steps: int = 10


class BeliefConfig(BaseModel):
    n_particles: int = 32
    transition_noise: float = 0.03
    latent_dim: int = 3
    likelihood_hidden_dims: list[int] = Field(default_factory=lambda: [32, 32])


class SearchConfig(BaseModel):
    enabled: bool = False
    n_rollouts: int = 16
    horizon: int = 4
    alpha_kl: float = 0.5
    kl_cap: float = 0.08


class LoggingConfig(BaseModel):
    tensorboard: bool = True
    run_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"


class AdvantageFilterConfig(BaseModel):
    enabled: bool = True
    quantile: float = 0.7
    min_abs_adv: float = 0.01


class AppConfig(BaseModel):
    seed: int = 7
    env: EnvConfig = Field(default_factory=EnvConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    belief: BeliefConfig = Field(default_factory=BeliefConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    advantage_filter: AdvantageFilterConfig = Field(default_factory=AdvantageFilterConfig)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    tensorboard_enabled: bool = True


def load_config(path: Path) -> AppConfig:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(raw)
