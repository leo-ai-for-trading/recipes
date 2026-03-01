import numpy as np

from polyagent.search.decision_time_search import improve_policy_distribution, kl_divergence


def test_search_distribution_is_normalized() -> None:
    pi_base = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)
    q = np.array([0.0, 1.0, -0.5, 0.2, 0.1, -0.1], dtype=np.float32)
    pi = improve_policy_distribution(pi_base=pi_base, q_values=q, alpha_kl=0.5, kl_cap=0.2)
    assert np.isclose(pi.sum(), 1.0)
    assert np.all(pi > 0)


def test_search_kl_is_capped() -> None:
    pi_base = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05], dtype=np.float32)
    q = np.array([0.0, 5.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    cap = 0.05
    pi = improve_policy_distribution(pi_base=pi_base, q_values=q, alpha_kl=0.1, kl_cap=cap)
    assert kl_divergence(pi, pi_base) <= cap + 1e-6
