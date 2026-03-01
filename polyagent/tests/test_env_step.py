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
