from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from polyalpha.config import AppSettings
from polyalpha.pipelines.compute_pnl import WalletPerformance

logger = logging.getLogger(__name__)

MEMO_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "wallet": {"type": "string"},
        "summary": {"type": "string"},
        "key_evidence": {"type": "array", "items": {"type": "string"}},
        "benign_explanations": {"type": "array", "items": {"type": "string"}},
        "followups": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["wallet", "summary", "key_evidence", "benign_explanations", "followups"],
}


@dataclass(slots=True)
class WalletMemo:
    wallet: str
    summary: str
    key_evidence: list[str]
    benign_explanations: list[str]
    followups: list[str]


def generate_memos(
    *,
    wallet_rows: list[WalletPerformance],
    settings: AppSettings,
) -> list[WalletMemo]:
    if not settings.openai_memo_enabled:
        return []
    if not settings.openai_api_key:
        logger.warning("OPENAI_API_KEY missing. Skipping memo generation.")
        return []

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed. Skipping memo generation.")
        return []

    client = OpenAI(api_key=settings.openai_api_key)
    top_rows = wallet_rows[: settings.openai_memo_top_n]
    memos: list[WalletMemo] = []

    for wallet in top_rows:
        payload = _wallet_prompt_payload(wallet)
        try:
            response = client.responses.create(
                model=settings.openai_model,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "You analyze public prediction-market wallet data. "
                            "Do not claim illicit intent; focus on observable patterns."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload)},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "wallet_memo",
                        "schema": MEMO_SCHEMA,
                        "strict": True,
                    }
                },
            )
            parsed = json.loads(response.output_text)
            memos.append(
                WalletMemo(
                    wallet=str(parsed.get("wallet", wallet.wallet_address)),
                    summary=str(parsed.get("summary", "")),
                    key_evidence=[str(x) for x in parsed.get("key_evidence", [])],
                    benign_explanations=[str(x) for x in parsed.get("benign_explanations", [])],
                    followups=[str(x) for x in parsed.get("followups", [])],
                )
            )
        except Exception as exc:  # pragma: no cover - network/API variability
            logger.warning("Memo generation failed for %s: %s", wallet.wallet_address, exc)

    return memos


def memos_to_rows(memos: list[WalletMemo]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for memo in memos:
        rows.append(
            {
                "Wallet Address": memo.wallet,
                "Summary": memo.summary,
                "Key Evidence": "\n".join(memo.key_evidence),
                "Benign Explanations": "\n".join(memo.benign_explanations),
                "Follow-ups": "\n".join(memo.followups),
            }
        )
    return rows


def _wallet_prompt_payload(wallet: WalletPerformance) -> dict[str, object]:
    return {
        "wallet": wallet.wallet_address,
        "name": wallet.name,
        "tags": wallet.tags,
        "wins": wallet.wins,
        "losses": wallet.losses,
        "win_rate": wallet.win_rate,
        "total_pnl": wallet.total_pnl,
        "realized_pnl": wallet.realized_pnl,
        "unrealized_pnl": wallet.unrealized_pnl,
        "total_invested": wallet.total_invested,
        "trade_count": wallet.trade_count,
        "top_markets": [
            {
                "condition_id": market.condition_id,
                "question": market.question,
                "total_pnl": market.total_pnl,
                "invested": market.invested,
            }
            for market in wallet.top_markets
        ],
    }
