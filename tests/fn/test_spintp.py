"""Tests for spintp.schabenberger_intensity_estimation (eq. 3.14, Diggle 1985)."""

import math

import pytest

from morie.fn.spintp import schabenberger_intensity_estimation


def _pts(n=30):
    return [[5 * (1 + math.sin(1.3 * k)), 5 * (1 + math.cos(0.7 * k + 0.4))] for k in range(n)]


def _k(t, kind):
    if kind == "gaussian":
        return math.exp(-0.5 * t * t) / math.sqrt(2 * math.pi)
    if abs(t) > 1:
        return 0.0
    return {"quadratic": 0.75 * (1 - t * t), "minimum_variance": 0.375 * (3 - 5 * t * t),
            "uniform": 0.5}[kind]


def _mass(x, lo, hi, h, kind):
    """p_h along one axis: integral over [lo, hi] of k((x-u)/h)/h du,
    by composite Simpson on 4000 panels (the kernels are polynomial or
    smooth on each panel, so this is exact to ~1e-12)."""
    m = 4000
    a, b = (x - hi) / h, (x - lo) / h
    if kind != "gaussian":
        a, b = max(a, -1.0), min(b, 1.0)
        if b <= a:
            return 0.0
    d = (b - a) / m
    s = _k(a, kind) + _k(b, kind) + sum((4 if i % 2 else 2) * _k(a + i * d, kind) for i in range(1, m))
    return s * d / 3


@pytest.mark.parametrize("kind", ["gaussian", "quadratic", "minimum_variance", "uniform"])
def test_spintp_basic(kind):
    """At every grid node lambda(s) = sum_i k((x-x_i)/hx) k((y-y_i)/hy)
    / (hx hy p_h(s)) with p_h(s) the kernel mass inside A, computed here
    by independent quadrature; integrated_intensity is the trapezoid
    rule over the grid."""
    P = _pts()
    reg = (0.0, 10.0, 0.0, 10.0)
    h = 2.5
    r = schabenberger_intensity_estimation(P, bandwidth=h, region=reg, grid=7, kernel=kind)
    xs = [float(v) for v in r["x_grid"]]
    ys = [float(v) for v in r["y_grid"]]
    lam = r["intensity_surface"].tolist()
    tot = 0.0
    for a, x in enumerate(xs):
        for b, y in enumerate(ys):
            raw = sum(_k((x - p[0]) / h, kind) * _k((y - p[1]) / h, kind) for p in P) / (h * h)
            ph = max(_mass(x, 0, 10, h, kind) * _mass(y, 0, 10, h, kind), 1e-6)
            assert lam[a][b] == pytest.approx(raw / ph, rel=1e-9, abs=1e-12)
            tot += (0.5 if a in (0, 6) else 1.0) * (0.5 if b in (0, 6) else 1.0) * lam[a][b]
    assert r["integrated_intensity"] == pytest.approx(tot * (10 / 6) ** 2, rel=1e-12)
    assert r["mean_intensity"] == pytest.approx(30 / 100, rel=1e-15)


def test_spintp_edge():
    """At a corner of A a symmetric kernel keeps exactly a quarter of its
    mass; without correction the weights are 1; bad inputs raise."""
    P = _pts()
    r = schabenberger_intensity_estimation(P, bandwidth=1.0, region=(0, 10, 0, 10), grid=5)
    assert r["edge_weight"].tolist()[0][0] == pytest.approx(0.25, abs=1e-15)
    r0 = schabenberger_intensity_estimation(P, bandwidth=1.0, region=(0, 10, 0, 10), grid=5, edge_correct=False)
    assert r0["edge_weight_min"] == 1.0
    with pytest.raises(ValueError):
        schabenberger_intensity_estimation([[1.0, 2.0, 3.0]] * 3)
    with pytest.raises(ValueError):
        schabenberger_intensity_estimation(P, kernel="triweight")
    with pytest.raises(ValueError):
        schabenberger_intensity_estimation(P, region=(1, 1, 0, 1))


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.spintp as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
