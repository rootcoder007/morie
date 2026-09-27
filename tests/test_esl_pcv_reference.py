"""Principal curves (Hastie-Stuetzle, ESL 14.61-14.62): self-consistency, a line recovered exactly, parity with R."""

import math

from morie.fn.eslgam import _spline_smooth
from morie.fn.eslpcv import esl_principal_curve


def arc():
    th = [k / 39 * math.pi for k in range(40)]
    return [[math.cos(t) + 0.05 * math.sin(7 * k), math.sin(t) + 0.05 * math.cos(5 * k)] for k, t in enumerate(th)]


def test_principal_curve_properties():
    X = arc()
    r = esl_principal_curve(X, penalty=0.05)
    p = r["distance_path"]
    assert p[0] > p[1] > p[2] > p[3]  # the first passes bend the PC line onto the arc
    assert p[0] > 4 and r["distance"] < 0.06
    # self-consistency: each coordinate of the curve is (nearly) the smooth of the data on lambda
    for j in range(2):
        sm = _spline_smooth(r["lambda"], [x[j] for x in X], 0.05)
        assert max(abs(a - f[j]) for a, f in zip(sm, r["fitted"])) < 0.02
    L = [[k / 10, 2 * k / 10 + 1] for k in range(20)]
    s = esl_principal_curve(L)
    assert s["distance"] < 1e-20 and s["converged"]  # a line is its own principal curve
    assert abs(r["distance"] - 0.054911640893359329) < 1e-9  # R arm, same algorithm
