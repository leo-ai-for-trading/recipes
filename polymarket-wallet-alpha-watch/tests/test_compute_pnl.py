from polyalpha.clients.data import PositionRecord
from polyalpha.pipelines.compute_pnl import aggregate_wallet_performance
from polyalpha.pipelines.discover_markets import DiscoveredMarket


def test_win_loss_and_win_rate_counting() -> None:
    closed = [
        PositionRecord.model_validate({"conditionId": "m1", "totalBought": 100, "realizedPnl": 25}),
        PositionRecord.model_validate({"conditionId": "m2", "totalBought": 70, "realizedPnl": -10}),
        PositionRecord.model_validate({"conditionId": "m3", "totalBought": 20, "realizedPnl": 0}),
    ]
    result = aggregate_wallet_performance(
        wallet_address="0xabc",
        closed_positions=closed,
        open_positions=[],
        topic_name="google",
        trade_count=8,
        profile=None,
        fallback_name=None,
        fallback_pseudonym=None,
        market_lookup={},
    )
    assert result.wins == 1
    assert result.losses == 1
    assert result.win_rate == 0.5
    assert result.realized_pnl == 15.0
    assert result.unrealized_pnl == 0.0
    assert result.total_pnl == 15.0


def test_pnl_aggregation_realized_and_unrealized() -> None:
    closed = [
        PositionRecord.model_validate({"conditionId": "m1", "totalBought": 20, "realizedPnl": 2}),
    ]
    opened = [
        PositionRecord.model_validate(
            {
                "conditionId": "m2",
                "totalBought": 30,
                "realizedPnl": 1,
                "cashPnl": 4,  # unrealized = cashPnl - realized = 3
            }
        ),
        PositionRecord.model_validate(
            {
                "conditionId": "m3",
                "totalBought": 10,
                "currentValue": 15,
                "initialValue": 10,  # unrealized = 5
            }
        ),
    ]
    markets = {
        "m1": DiscoveredMarket(condition_id="m1", question="Q1"),
        "m2": DiscoveredMarket(condition_id="m2", question="Q2"),
    }
    result = aggregate_wallet_performance(
        wallet_address="0xdef",
        closed_positions=closed,
        open_positions=opened,
        topic_name="google",
        trade_count=10,
        profile=None,
        fallback_name="WalletX",
        fallback_pseudonym=None,
        market_lookup=markets,
    )
    assert result.wins == 1
    assert result.losses == 0
    assert result.realized_pnl == 3.0
    assert result.unrealized_pnl == 8.0
    assert result.total_pnl == 11.0
    assert result.total_invested == 60.0
    assert result.name == "WalletX"
    assert "google" in result.tags
    assert len(result.top_markets) == 3


def test_empty_wallet_positions_returns_zero_metrics() -> None:
    result = aggregate_wallet_performance(
        wallet_address="0xempty",
        closed_positions=[],
        open_positions=[],
        topic_name="google",
        trade_count=0,
        profile=None,
        fallback_name=None,
        fallback_pseudonym=None,
        market_lookup={},
    )
    assert result.wins == 0
    assert result.losses == 0
    assert result.win_rate == 0.0
    assert result.realized_pnl == 0.0
    assert result.unrealized_pnl == 0.0
    assert result.total_pnl == 0.0
    assert result.total_invested == 0.0
