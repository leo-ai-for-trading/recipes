from __future__ import annotations

from pydantic import BaseModel


class GammaTag(BaseModel):
    id: int
    slug: str | None = None
    label: str | None = None


class GammaClient:
    """Gamma API client placeholder (implemented in Step B)."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
