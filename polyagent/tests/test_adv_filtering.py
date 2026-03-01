import numpy as np

from polyagent.rl.utils import advantage_filter_mask


def test_advantage_filter_quantile_keeps_top_signal() -> None:
    adv = np.array([0.1, -0.2, 0.01, 1.4, -1.0], dtype=np.float32)
    mask = advantage_filter_mask(
        adv,
        enabled=True,
        quantile=0.7,
        min_abs_adv=0.0,
    )
    assert mask.dtype == bool
    assert mask.sum() >= 2
    assert mask[3]
    assert mask[4]


def test_advantage_filter_min_abs_threshold() -> None:
    adv = np.array([0.01, -0.03, 0.2, -0.4], dtype=np.float32)
    mask = advantage_filter_mask(
        adv,
        enabled=True,
        quantile=0.0,
        min_abs_adv=0.1,
    )
    assert np.array_equal(mask, np.array([False, False, True, True]))


def test_advantage_filter_disabled_keeps_all() -> None:
    adv = np.array([0.01, -0.03, 0.2, -0.4], dtype=np.float32)
    mask = advantage_filter_mask(
        adv,
        enabled=False,
        quantile=0.9,
        min_abs_adv=10.0,
    )
    assert mask.all()
