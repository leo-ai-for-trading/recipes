from pathlib import Path

from polyagent.config import apply_dot_overrides, load_config


def test_apply_dot_overrides_updates_nested_values() -> None:
    cfg = load_config(Path("configs/mvp.yaml"))
    improved = apply_dot_overrides(
        cfg,
        {
            "rl.kl_target": 0.005,
            "search.enabled": False,
            "env.inventory_penalty": 0.002,
        },
    )
    assert improved.rl.kl_target == 0.005
    assert improved.search.enabled is False
    assert improved.env.inventory_penalty == 0.002
