"""Tests for ranova (ANOVA variance components, one-way random model).

Replaces the generated stub, which imported a name the module never had.
"""

from morie.fn.ranova import ranova


def _balanced():
    # three classes of four, with a clear between-class spread
    y, g = [], []
    for i, base in enumerate((10.0, 20.0, 30.0)):
        for k, off in enumerate((-1.0, 0.0, 1.0, 0.0)):
            y.append(base + off)
            g.append(i)
    return y, g


def test_the_sums_of_squares_add_up():
    y, g = _balanced()
    res = ranova(y, g)
    grand = sum(y) / len(y)
    total = sum((v - grand) ** 2 for v in y)
    assert abs(res["ssa"] + res["sse"] - total) < 1e-9
    assert res["a"] == 3
    assert res["balanced"] is True


def test_the_mean_squares_are_the_sums_over_their_df():
    y, g = _balanced()
    res = ranova(y, g)
    assert abs(res["msa"] - res["ssa"] / (res["a"] - 1)) < 1e-9
    assert abs(res["mse"] - res["sse"] / (len(y) - res["a"])) < 1e-9


def test_the_moment_estimator_is_msa_minus_mse_over_n0():
    y, g = _balanced()
    res = ranova(y, g)
    want = (res["msa"] - res["mse"]) / res["n0"]
    assert abs(res["sigma2_a_raw"] - want) < 1e-9
    assert abs(res["n0"] - 4.0) < 1e-9        # balanced, four per class


def test_a_negative_component_is_truncated_but_reported_raw():
    # no between-class signal at all: the moment estimate can go negative
    y = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    g = [0, 1, 0, 1, 0, 1, 0, 1]
    res = ranova(y, g)
    assert res["sigma2_a"] >= 0.0
    assert res["sigma2_a_raw"] <= res["sigma2_a"] + 1e-12


def test_the_icc_is_the_share_of_the_between_class_variance():
    y, g = _balanced()
    res = ranova(y, g)
    want = res["sigma2_a"] / (res["sigma2_a"] + res["sigma2_e"])
    assert abs(res["icc"] - want) < 1e-12
    assert res["icc"] > 0.9        # classes 10, 20, 30 are far apart


def test_unbalanced_classes_are_handled_and_flagged():
    y = [1.0, 2.0, 3.0, 10.0, 11.0]
    g = [0, 0, 0, 1, 1]
    res = ranova(y, g)
    assert res["balanced"] is False
    assert res["n_i"] == [3, 2]


def test_validation():
    for call in (lambda: ranova([1.0, 2.0], [0]),
                 lambda: ranova([1.0, 2.0], [0, 0]),
                 lambda: ranova([1.0, 2.0], [0, 1])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
