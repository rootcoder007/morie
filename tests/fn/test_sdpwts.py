"""Tests for sdpwts.semidefinite_program (barrier method, B&V sec. 11.6)."""

import pytest

from morie.fn.sdpwts import min_eigenvalue_sdp, semidefinite_program


F0 = [[0.0, 1.0], [1.0, 0.0]]
I2 = [[1.0, 0.0], [0.0, 1.0]]


def test_sdpwts_basic():
    """min x s.t. F0 + x I >= 0 has x* = -lambda_min(F0) = 1; on the
    central path the duality gap is exactly m / t, so the returned point
    satisfies 0 <= c'x - p* <= gap."""
    r = semidefinite_program([1.0], F0, [I2], [2.0])
    assert r["gap"] == pytest.approx(r["m"] / r["path"][-1]["t"], rel=1e-15)
    assert r["gap"] < 1e-8
    assert 0.0 <= r["objective"] - 1.0 <= r["gap"]
    assert r["min_eigenvalue"] > 0.0
    e = min_eigenvalue_sdp([[3.0, 1.0], [1.0, 2.0]])
    assert e["lambda_min"] == pytest.approx((5 - 5 ** 0.5) / 2, rel=1e-12)
    assert e["error"] <= e["gap"]


def test_sdpwts_edge():
    """An infeasible start and mu <= 1 raise."""
    with pytest.raises(ValueError):
        semidefinite_program([1.0], F0, [I2], [0.5])
    with pytest.raises(ValueError):
        semidefinite_program([1.0], F0, [I2], [2.0], mu=1.0)
