"""Tests for btarsv (AR-sieve bootstrap, Buhlmann 1997).

Replaces the generated stub, which imported ``boot_ar_sieve``.
"""

import math

from morie.fn.btarsv import btarsv


def _ar1(phi=0.6, n=300, seed=5):
    st = [seed]

    def r():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)

    x, prev = [], 0.0
    for _ in range(n + 100):
        e = math.sqrt(-2 * math.log(max(r(), 1e-12))) * \
            math.cos(2 * math.pi * r())
        prev = phi * prev + e
        x.append(prev)
    return x[100:]


def test_the_fitted_ar_coefficient_recovers_the_truth():
    res = btarsv(_ar1(phi=0.6), p=1, B=50, seed=1)
    assert abs(res["phi"][0] - 0.6) < 0.12
    assert res["sigma2"] > 0


def test_order_selection_picks_a_small_order_for_ar1():
    res = btarsv(_ar1(phi=0.6), B=30, seed=1, p_max=8)
    assert 1 <= res["p"] <= 4


def test_residuals_are_centred_and_replicates_are_returned():
    res = btarsv(_ar1(), p=1, B=60, seed=2)
    assert abs(res["residual_mean"]) < 1e-9
    assert len(res["replicates"]) == 60
    assert res["se"] > 0


def test_a_stronger_dependence_inflates_the_standard_error():
    weak = btarsv(_ar1(phi=0.1), p=1, B=80, seed=3)["se"]
    strong = btarsv(_ar1(phi=0.85), p=1, B=80, seed=3)["se"]
    assert strong > weak


def test_seed_reproducibility():
    a = btarsv(_ar1(), p=1, B=40, seed=9)["replicates"]
    b = btarsv(_ar1(), p=1, B=40, seed=9)["replicates"]
    assert a == b


def test_validation():
    for call in (lambda: btarsv([1.0] * 10),
                 lambda: btarsv([1.0] * 50),
                 lambda: btarsv(_ar1(), p=999)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
