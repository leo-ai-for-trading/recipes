from __future__ import annotations


def compute_fee(notional: float, fee_bps: float) -> float:
    return abs(notional) * (fee_bps / 10_000.0)
