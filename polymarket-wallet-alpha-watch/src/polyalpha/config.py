from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TopicConfig(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    tag_slugs: list[str] = Field(default_factory=list)
    tag_ids: list[int] = Field(default_factory=list)
    include_related_tags: bool = True
    include_closed_markets: bool = True


class TopicsConfig(BaseModel):
    topics: dict[str, TopicConfig]


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gamma_base_url: str = "https://gamma-api.polymarket.com"
    data_base_url: str = "https://data-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"

    topics_config_path: Path = Path("config/topics.yaml")

    request_timeout_seconds: float = 20.0
    request_max_retries: int = 5
    request_backoff_seconds: float = 0.5
    max_concurrency: int = 8
    requests_per_second: float = 8.0

    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_memo_enabled: bool = False
    openai_memo_top_n: int = 25


def load_topics(path: Path) -> TopicsConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TopicsConfig.model_validate(raw)


def get_topic_or_raise(config: TopicsConfig, topic_name: str) -> TopicConfig:
    topic = config.topics.get(topic_name)
    if topic is None:
        available = ", ".join(sorted(config.topics))
        raise ValueError(f"Unknown topic '{topic_name}'. Available: {available}")
    return topic
