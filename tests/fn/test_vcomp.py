"""Tests for vcomp (variance components with an ICC interval).

Replaces the generated stub, which imported
``variance_components_henderson3``.
"""

from morie.fn.vcomp import vcomp


def _balanced(a=6, n=5, spread=4.0):
    y, g = [], []
    for i in range(a):
        for k in range(n):
            y.append(spread * i + (k - (n - 1) / 2.0))
            g.append(i)
    return y, g


def test_both_methods_run_and_broadly_agree():
    y, g = _balanced()
    reml = vcomp(y, g, method="reml")
    anova = vcomp(y, g, method="anova")
    assert abs(reml["icc"] - anova["icc"]) < 0.1


def test_the_icc_interval_brackets_the_estimate():
    y, g = _balanced()
    res = vcomp(y, g)
    assert res["icc_lower"] <= res["icc"] <= res["icc_upper"]
    assert 0.0 <= res["icc_lower"] and res["icc_upper"] <= 1.0


def test_a_wider_confidence_level_widens_the_interval():
    y, g = _balanced()
    narrow = vcomp(y, g, conf_level=0.80)
    wide = vcomp(y, g, conf_level=0.99)
    assert (wide["icc_upper"] - wide["icc_lower"]) > \
        (narrow["icc_upper"] - narrow["icc_lower"])


def test_no_class_signal_gives_an_icc_near_zero():
    y = [1.0, 2.0, 3.0] * 5
    g = sum(([i] * 3 for i in range(5)), [])
    res = vcomp(y, g)
    assert res["icc"] < 0.05


def test_validation():
    y, g = _balanced()
    for call in (lambda: vcomp([1.0, 2.0], [0, 1]),
                 lambda: vcomp(y, g, method="ml"),
                 lambda: vcomp(y, g, conf_level=1.5)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
