"""Tests for shscl.scaled_schoenfeld_residual."""

import math

import pytest

from morie.fn.shscl import scaled_schoenfeld_residual


N = 40
X1 = [((i * 7) % 11) / 5 - 1 for i in range(N)]
X2 = [math.cos(i) for i in range(N)]
TIME = [round(3 + ((i * 13) % 17) / 2 + 0.7 * a + 0.3 * math.sin(i), 3) for i, a in enumerate(X1)]
EVENT = [0 if (i * 5) % 7 == 0 else 1 for i in range(N)]
X = [[a, b] for a, b in zip(X1, X2)]


def test_shscl_basic():
    """Statistics equal R survival 3.x cox.zph(coxph(..., ties = "breslow"),
    terms = FALSE) under the Kaplan-Meier and rank transforms (tied times
    and censoring included)."""
    r = scaled_schoenfeld_residual(TIME, EVENT, X, transform="km")
    assert r["statistic"] == pytest.approx([0.47916647734089746, 0.8249263749171708], rel=1e-9)
    assert r["global_statistic"] == pytest.approx(1.2360592061885678, rel=1e-9)
    q = scaled_schoenfeld_residual(TIME, EVENT, X, transform="rank")
    assert q["statistic"] == pytest.approx([0.464436490974312, 0.685375577344253], rel=1e-9)
    assert q["global_statistic"] == pytest.approx(1.10549294759634, rel=1e-9)


def test_shscl_edge():
    """The scaled residuals average to beta-hat (the score equations), and
    match cox.zph's first three for x1."""
    r = scaled_schoenfeld_residual(TIME, EVENT, X)
    d = len(r["scaled"])
    for k in range(2):
        assert sum(row[k] for row in r["scaled"]) / d == pytest.approx(r["beta"][k], abs=1e-9)
    assert [row[0] for row in r["scaled"][:3]] == pytest.approx(
        [-1.9526065958053183, 2.1756979662607723, 0.64193251674218321], rel=1e-9)
    with pytest.raises(ValueError, match="transform"):
        scaled_schoenfeld_residual(TIME, EVENT, X, transform="sqrt")


