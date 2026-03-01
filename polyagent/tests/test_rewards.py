from polyagent.env.fees import compute_fee
from polyagent.env.fills import simulate_fill


def test_fee_is_positive_and_symmetric() -> None:
    fee1 = compute_fee(1000.0, 2.0)
    fee2 = compute_fee(-1000.0, 2.0)
    assert fee1 > 0
    assert fee1 == fee2


def test_maker_fill_logic_buy_at_best_bid() -> None:
    fill = simulate_fill(
        action=1,
        order_size=10.0,
        best_bid=0.48,
        best_ask=0.52,
        next_trade_price=0.48,
        next_trade_side=-1,
        tick_size=0.01,
    )
    assert fill.filled_qty == 10.0
    assert fill.is_maker is True


def test_taker_fill_logic_buy_crosses_spread() -> None:
    fill = simulate_fill(
        action=2,
        order_size=5.0,
        best_bid=0.48,
        best_ask=0.52,
        next_trade_price=0.0,
        next_trade_side=0,
        tick_size=0.01,
    )
    assert fill.filled_qty == 5.0
    assert fill.fill_price == 0.52
    assert fill.is_maker is False
