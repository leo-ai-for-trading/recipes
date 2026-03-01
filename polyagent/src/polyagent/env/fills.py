from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FillResult:
    filled_qty: float
    fill_price: float
    is_maker: bool


def simulate_fill(
    *,
    action: int,
    order_size: float,
    best_bid: float,
    best_ask: float,
    next_trade_price: float,
    next_trade_side: int,
    tick_size: float,
) -> FillResult:
    if action == 0 or action == 5:
        return FillResult(filled_qty=0.0, fill_price=0.0, is_maker=True)

    # 1) Bid at best bid -> maker fill only if next trade is sell and reaches bid.
    if action == 1:
        price = best_bid
        filled = order_size if (next_trade_side < 0 and next_trade_price <= price) else 0.0
        return FillResult(filled_qty=filled, fill_price=price if filled else 0.0, is_maker=True)

    # 2) Bid one tick up -> taker-style immediate cross at best ask.
    if action == 2:
        return FillResult(filled_qty=order_size, fill_price=best_ask, is_maker=False)

    # 3) Ask at best ask -> maker fill only if next trade is buy and reaches ask.
    if action == 3:
        price = best_ask
        filled = order_size if (next_trade_side > 0 and next_trade_price >= price) else 0.0
        return FillResult(filled_qty=-filled, fill_price=price if filled else 0.0, is_maker=True)

    # 4) Ask one tick down -> taker-style immediate cross at best bid.
    if action == 4:
        return FillResult(filled_qty=-order_size, fill_price=best_bid, is_maker=False)

    _ = tick_size
    return FillResult(filled_qty=0.0, fill_price=0.0, is_maker=True)
