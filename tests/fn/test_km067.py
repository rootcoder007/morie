"""Tests for km067.kamath_ch5_rm_bradley_terry."""

import math

from morie.fn import _array_core as np

from morie.fn.km067 import kamath_ch5_rm_bradley_terry


def test_km067_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n = 10
    x = [f"prompt_{i}" for i in range(n)]
    y_w = ["w"] * n
    y_l = ["l"] * n
    # Per-pair reward: 1.0 for the winning answer, 0.0 for the losing one.
    r_theta = lambda xi, yi: {"w": 1.0, "l": 0.0}[yi]
    out = kamath_ch5_rm_bradley_terry(x, y_w, y_l, r_theta)
    assert isinstance(out, dict)
    assert "estimate" in out
    assert "margins" in out
    assert "per_pair" in out
    assert "n" in out
    assert "method" in out
    assert out["n"] == n
    assert len(out["per_pair"]) == n
    assert math.isfinite(out["estimate"])
    # When r(x,y_w)-r(x,y_l)=1, every per-pair loss is log(1+exp(-1)).
    expected = math.log(1 + math.exp(-1))
    for v in out["per_pair"]:
        assert math.isfinite(v)
    assert abs(out["estimate"] - expected) < 1e-12


def test_km067_edge():
    """Test edge cases."""
    # Single-pair case, matching the docstring example.
    r_theta = lambda xi, yi: {"w": 1.0, "l": 0.0}[yi]
    out = kamath_ch5_rm_bradley_terry(["p"], ["w"], ["l"], r_theta)
    assert isinstance(out, dict)
    assert out["n"] == 1
    assert len(out["per_pair"]) == 1
    assert math.isfinite(out["estimate"])
    expected = math.log(1 + math.exp(-1))
    assert abs(out["estimate"] - expected) < 1e-12
    # A non-trivial reward function on a few pairs.
    rng = np.random.default_rng(7)
    n = 5
    xs = [f"q{i}" for i in range(n)]
    ys_w = ["w"] * n
    ys_l = ["l"] * n
    rewards = {"p0": 0.3, "p1": 0.8, "p2": 0.0, "p3": 1.2, "p4": -0.5,
               "q0": 0.3, "q1": 0.8, "q2": 0.0, "q3": 1.2, "q4": -0.5}
    r_theta2 = lambda xi, yi: rewards[xi] if yi == "w" else 0.0
    out2 = kamath_ch5_rm_bradley_terry(xs, ys_w, ys_l, r_theta2)
    assert isinstance(out2, dict)
    assert out2["n"] == n
    assert len(out2["per_pair"]) == n
    assert math.isfinite(out2["estimate"])
