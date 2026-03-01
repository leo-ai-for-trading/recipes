from pathlib import Path

from polyagent.config import EnvConfig
from polyagent.data.loaders import load_market_data
from polyagent.data.make_demo_data import make_demo_data
from polyagent.env.replay_env import ReplayEnv


def test_replay_env_step(tmp_path: Path) -> None:
    data_path = make_demo_data(tmp_path / "demo.parquet", n_steps=120, seed=3)
    df = load_market_data(data_path)
    env = ReplayEnv(df, EnvConfig())
    obs, info = env.reset(seed=7)
    assert obs.shape == (20,)
    assert "inventory" in info

    next_obs, reward, done, truncated, step_info = env.step(0)
    assert next_obs.shape == (20,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert "mtm_pnl" in step_info


def test_inventory_guardrail_blocks_excess_buy(tmp_path: Path) -> None:
    data_path = make_demo_data(tmp_path / "demo.parquet", n_steps=120, seed=3)
    df = load_market_data(data_path)
    cfg = EnvConfig(order_size=10.0, max_inventory=5.0)
    env = ReplayEnv(df, cfg)
    _, _ = env.reset(seed=7)
    _, _, _, _, info = env.step(2)  # immediate buy taker, inventory becomes +10
    assert info["inventory"] == 10.0
    _, _, _, _, info2 = env.step(2)  # should be blocked by max inventory guardrail
    assert info2["requested_action"] == 2
    assert info2["executed_action"] == 0
