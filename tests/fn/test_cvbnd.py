"""Tests for morie.fn.cvbnd — cross-validation risk bound."""

import pytest

from morie.fn import _array_core as np
from morie.fn.cvbnd import cvbnd


def test_basic_output():
    risks = np.array([0.5, 0.6, 0.55, 0.52, 0.58])
    result = cvbnd(risks, n=200)
    assert "cv_risk" in result
    assert "upper_bound" in result


def test_upper_bound_ge_risk():
    risks = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    result = cvbnd(risks, n=100)
    assert result["upper_bound"] >= result["cv_risk"]


def test_more_data_tighter():
    risks = np.array([0.5, 0.6, 0.55])
    r1 = cvbnd(risks, n=100)
    r2 = cvbnd(risks, n=10000)
    assert r2["bound_width"] < r1["bound_width"]


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        cvbnd(np.array([]), n=100)


def test_hoeffding_width_uses_the_loss_bound():
    import math

    risks = [0.21, 0.35, 0.28, 0.30, 0.25]
    r = cvbnd(risks, n=400, delta=0.1, loss_bound=2.0)
    w = 2.0 * math.sqrt(math.log(2 / 0.1) / (2 * 400))
    m = sum(risks) / 5
    assert r["bound_width"] == pytest.approx(w, rel=1e-14)
    assert r["upper_bound"] == pytest.approx(m + w, rel=1e-14)
    assert r["cv_se"] == pytest.approx(math.sqrt(sum((v - m) ** 2 for v in risks) / 4 / 5), rel=1e-14)
    # the width does not depend on how the fold risks happen to scatter
    assert cvbnd([0.3] * 5, n=400, delta=0.1, loss_bound=2.0)["bound_width"] == pytest.approx(w, rel=1e-14)
