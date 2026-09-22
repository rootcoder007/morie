"""Tests for fanmd.fanova_decomposition."""

from morie.fn import _array_core as np

from morie.fn.fanmd import fanova_decomposition


def test_fanmd_basic():
    """Test basic functionality on a simple additive function."""
    g = 8
    pts = [(t + 0.5) / g for t in range(g)]

    def f(x):
        # f(x1, x2) = x1 + 2*x2  (purely additive: only first-order effects)
        return x[0] + 2.0 * x[1]

    # input_dist is None: tf returns row unchanged -> f sees raw grid coordinates.
    res = fanova_decomposition(f, input_dist=None, d=2, grid=g)

    # Returned as RichResult; payload holds the data dict.
    payload = res.payload

    # Result must expose the documented keys.
    assert "estimate" in payload
    assert "D" in payload
    assert "D_main" in payload
    assert "D_int" in payload
    assert "closure" in payload

    f0 = payload["estimate"]
    D = payload["D"]
    D_main = payload["D_main"]
    D_int = payload["D_int"]

    # Shapes: d first-order components, d*(d-1)/2 pairwise components.
    assert len(D_main) == 2
    assert len(D_int) == 1

    # For an additive function, all interaction variance must be zero.
    assert D_int[0] == 0.0

    # All values non-negative.
    assert D >= 0.0
    for v in D_main:
        assert v >= 0.0
    for v in D_int:
        assert v >= 0.0

    # Closure should equal 1.0 for a two-dimensional additive decomposition
    # (Hoeffding/Sobol: D == D_main_1 + D_main_2 + D_int).
    assert abs(payload["closure"] - 1.0) < 1e-12

    # Sanity check against an independent arithmetic implementation.
    # For f(x1, x2) = x1 + 2 x2 with x on the same tensor grid points pts:
    #   f0  = E[x1] + 2 E[x2]   (E[pts] = (g - 1)/(2g) -> wait, pts are (t+0.5)/g,
    #         mean of uniform centres = (g + 1)/(2g) .. compute numerically below)
    # Build f0 and D from scratch on the same grid and compare.
    pts_mean = sum(pts) / g  # mean of the grid centres
    f0_ref = pts_mean + 2.0 * pts_mean  # = 3 * pts_mean
    # Var of pts on the centres (t+0.5)/g for t=0..g-1.
    pts_var = sum((p - pts_mean) ** 2 for p in pts) / g
    # f = x1 + 2 x2 over independent uniform grid -> Var(f) = 1*Var(x1) + 4*Var(x2)
    D_ref = 1.0 * pts_var + 4.0 * pts_var

    assert abs(f0 - f0_ref) < 1e-12
    assert abs(D - D_ref) < 1e-12

    # Main-effect variances for an additive function equal Var of each component.
    D_main_ref = [pts_var, 4.0 * pts_var]
    for got, want in zip(D_main, D_main_ref):
        assert abs(got - want) < 1e-12


def test_fanmd_edge():
    """Test edge cases: scalar (d=1) and constant function."""
    g = 8
    pts = [(t + 0.5) / g for t in range(g)]

    def constant_f(x):
        return 7.5

    # d=1: no pairwise interactions.
    res1 = fanova_decomposition(constant_f, input_dist=None, d=1, grid=g)
    p1 = res1.payload

    # f0 equals the constant value; total variance is zero.
    assert abs(p1["estimate"] - 7.5) < 1e-12
    assert p1["D"] == 0.0
    assert len(p1["D_main"]) == 1
    assert len(p1["D_int"]) == 0
    # With zero total variance, closure is NaN (formula returns nan in that case).
    assert p1["closure"] != p1["closure"]  # NaN != NaN

    # d=2 constant: closure is NaN (D == 0).
    def const2(x):
        return 3.0

    res2 = fanova_decomposition(const2, input_dist=None, d=2, grid=g)
    p2 = res2.payload
    assert abs(p2["estimate"] - 3.0) < 1e-12
    assert p2["D"] == 0.0
    assert len(p2["D_main"]) == 2
    assert len(p2["D_int"]) == 1
    assert p2["D_int"][0] == 0.0
    assert p2["closure"] != p2["closure"]  # NaN
