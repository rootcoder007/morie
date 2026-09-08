"""Tests for remlfn (REML variance components, one-way random model).

Replaces the generated stub, which imported a name the module never had.
"""

from morie.fn.remlfn import remlfn


def _balanced(a=6, n=5, spread=5.0):
    y, g = [], []
    for i in range(a):
        base = spread * i
        for k in range(n):
            y.append(base + (k - (n - 1) / 2.0))
            g.append(i)
    return y, g


def test_reml_agrees_with_the_closed_form_when_balanced():
    y, g = _balanced()
    closed = remlfn(y, g, solver="closed")
    optim = remlfn(y, g, solver="optim")
    assert abs(closed["sigma2_a"] - optim["sigma2_a"]) < 1e-4
    assert abs(closed["sigma2_e"] - optim["sigma2_e"]) < 1e-6
    assert closed["closed_form"] is True


def test_the_grand_mean_is_estimated():
    y, g = _balanced()
    res = remlfn(y, g)
    assert abs(res["mu"] - sum(y) / len(y)) < 1e-6


def test_components_are_non_negative_and_the_icc_follows_them():
    y, g = _balanced()
    res = remlfn(y, g)
    assert res["sigma2_a"] >= 0 and res["sigma2_e"] > 0
    want = res["sigma2_a"] / (res["sigma2_a"] + res["sigma2_e"])
    assert abs(res["icc"] - want) < 1e-9
    assert res["icc"] > 0.8


def test_no_between_class_signal_gives_a_near_zero_component():
    y = [1.0, 2.0, 3.0] * 4
    g = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    res = remlfn(y, g)
    assert res["sigma2_a"] < 1e-6
    assert res["icc"] < 1e-6


def test_it_converges_and_says_so():
    y, g = _balanced()
    res = remlfn(y, g, solver="optim")
    assert res["converged"] is True
    assert res["loglik"] == res["loglik"]      # not NaN


def test_validation():
    y, g = _balanced()
    for call in (lambda: remlfn([1.0, 2.0], [0]),
                 lambda: remlfn([1.0, 2.0], [0, 0]),
                 lambda: remlfn([1.0, 2.0], [0, 1]),
                 lambda: remlfn(y, g, solver="em")):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
