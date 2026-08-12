"""Tests for btiseq (iterated prepivoted bootstrap test, Beran 1988).

Replaces the generated stub, which imported ``boot_iter_calibrated``.
"""

from morie.fn.btiseq import btiseq


def _sample(mu=0.0, n=60, seed=6):
    st = [seed]

    def r():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return [mu + 3.0 * (r() - 0.5) for _ in range(n)]


def test_a_true_null_is_not_rejected():
    res = btiseq(_sample(mu=0.0), mu0=0.0, B_outer=200, B_inner=80,
                 seed=1)
    assert res["p_boot"] > 0.05
    assert res["p_iterated"] > 0.05


def test_a_far_null_is_rejected():
    res = btiseq(_sample(mu=0.0), mu0=5.0, B_outer=200, B_inner=80,
                 seed=1)
    assert res["p_boot"] < 0.05
    assert res["p_iterated"] < 0.05


def test_p_values_are_probabilities_and_the_statistic_is_reported():
    res = btiseq(_sample(), mu0=0.3, B_outer=150, B_inner=60, seed=1)
    assert 0.0 <= res["p_boot"] <= 1.0
    assert 0.0 <= res["p_iterated"] <= 1.0
    assert 0.0 <= res["prepivoted_value"] <= 1.0
    assert res["mu0"] == 0.3


def test_seed_reproducibility():
    x = _sample()
    a = btiseq(x, mu0=0.2, B_outer=100, B_inner=50, seed=5)
    b = btiseq(x, mu0=0.2, B_outer=100, B_inner=50, seed=5)
    assert a["p_boot"] == b["p_boot"]
    assert a["p_iterated"] == b["p_iterated"]


def test_validation():
    try:
        btiseq([1.0, 2.0])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
