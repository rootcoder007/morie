"""Tests for caustrnsp (transporting a trial effect to a target).

Replaces the generated stub, which imported
``causal_transportability_weights``.
"""

from morie.fn.caustrnsp import caustrnsp


def _trial(n=200, effect=3.0):
    # sampling score varies with an index so the trial over-represents
    # the low-score half
    y, z, s = [], [], []
    for i in range(n):
        score = 0.2 + 0.6 * (i / float(n - 1))
        treated = i % 2
        base = 10.0 * score
        y.append(base + (effect * score * 2.0 if treated else 0.0))
        z.append(treated)
        s.append(score)
    return y, z, s


def test_the_weighted_arms_are_reported():
    y, z, s = _trial()
    res = caustrnsp(y, z, s)
    assert res["n"] == len(y)
    assert res["n_treat"] + res["n_control"] == len(y)
    assert abs(res["estimate"] -
               (res["mean_treated"] - res["mean_control"])) < 1e-9


def test_weights_are_the_inverse_odds_for_transport():
    y, z, s = _trial(n=20)
    res = caustrnsp(y, z, s, mode="transport")
    for i in range(len(y)):
        assert abs(res["weights"][i] - (1.0 - s[i]) / s[i]) < 1e-9


def test_generalize_weights_are_the_inverse_score():
    y, z, s = _trial(n=20)
    res = caustrnsp(y, z, s, mode="generalize")
    for i in range(len(y)):
        assert abs(res["weights"][i] - 1.0 / s[i]) < 1e-9
    assert res["mode"] == "generalize"


def test_a_constant_score_makes_weighting_a_no_op():
    # with every unit equally likely to be sampled, the weighted
    # difference is the unweighted one
    y = [1.0, 2.0, 3.0, 8.0, 9.0, 10.0]
    z = [0, 0, 0, 1, 1, 1]
    s = [0.5] * 6
    res = caustrnsp(y, z, s)
    assert abs(res["estimate"] - (9.0 - 2.0)) < 1e-9


def test_transport_and_generalize_differ_when_the_score_varies():
    y, z, s = _trial()
    a = caustrnsp(y, z, s, mode="transport")["estimate"]
    b = caustrnsp(y, z, s, mode="generalize")["estimate"]
    assert abs(a - b) > 1e-6


def test_validation():
    y, z, s = _trial(n=20)
    for call in (lambda: caustrnsp(y[:-1], z, s),
                 lambda: caustrnsp(y, [2] * len(y), s),
                 lambda: caustrnsp(y, z, [0.0] * len(y)),
                 lambda: caustrnsp(y, z, [1.0] * len(y)),
                 lambda: caustrnsp(y, z, s, mode="extrapolate")):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
