"""Tests for deepSVDD (support vector data description, Tax & Duin 2004).

Replaces the generated stub, which imported ``deep_svdd``; the module's
function is ``svdd``.
"""

import math

from morie.fn.deepSVDD import svdd


def _ring(n=24, r=2.0, cx=1.0, cy=-1.0):
    return [[cx + r * math.cos(2 * math.pi * i / n),
             cy + r * math.sin(2 * math.pi * i / n)] for i in range(n)]


def test_points_on_a_circle_give_back_its_centre_and_radius():
    res = svdd(_ring(), C=1.0)
    assert abs(res["center"][0] - 1.0) < 1e-6
    assert abs(res["center"][1] + 1.0) < 1e-6
    assert abs(math.sqrt(res["radius2"]) - 2.0) < 1e-6


def test_the_dual_weights_are_a_probability_vector():
    res = svdd(_ring(), C=1.0)
    assert abs(sum(res["alpha"]) - 1.0) < 1e-8
    assert all(a >= -1e-12 for a in res["alpha"])


def test_interior_points_are_not_support_vectors():
    X = _ring() + [[1.0, -1.0], [1.2, -0.9]]     # two interior points
    res = svdd(X, C=1.0)
    assert len(X) - 2 not in res["support"]
    assert len(X) - 1 not in res["support"]


def test_C_controls_whether_points_may_be_left_outside():
    X = _ring() + [[20.0, 20.0]]
    # C = 1 puts no cap on a single point's weight, so the sphere must
    # enclose everything and nothing is called an outlier
    assert svdd(X, C=1.0)["outliers"] == []
    # a small C caps each weight, and the fit starts leaving points out
    assert svdd(X, C=0.2)["outliers"]


def test_the_rbf_kernel_runs_and_keeps_the_kkt_conditions():
    res = svdd(_ring(), C=1.0, kernel="rbf", gamma=0.5)
    assert res["kernel"] == "rbf"
    assert res["kkt_violation"] < 1e-4


def test_validation():
    for call in (lambda: svdd([[1.0, 2.0]]),
                 lambda: svdd(_ring(), C=0.001),
                 lambda: svdd(_ring(), kernel="poly")):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
