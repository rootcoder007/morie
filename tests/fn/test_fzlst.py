"""Tests for fzlst.fauzi_l_statistic."""

import numpy as np
import pytest

from morie.fn.fzlst import fauzi_l_statistic


def test_fzlst_constant_score_gives_the_mean():
    """With J = 1 the L-statistic integrates the quantile function --
    i.e. the sample mean (up to quadrature error)."""
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 1.0, 400)
    r = fauzi_l_statistic(x)
    assert float(r["estimate"]) == pytest.approx(float(x.mean()), abs=0.02)


def test_fzlst_symmetric_score_weights_the_tails():
    """J(u) = 12(u - 1/2) is orthogonal to location: on symmetric data
    the statistic measures spread, so scaling x by 3 scales it by 3."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    J = lambda u: 12.0 * (u - 0.5)
    a = float(fauzi_l_statistic(x, score=J)["estimate"])
    b = float(fauzi_l_statistic(3.0 * x, score=J)["estimate"])
    assert b == pytest.approx(3.0 * a, rel=0.05)


def test_fzlst_location_shift_moves_the_mean_functional():
    rng = np.random.default_rng(2)
    x = rng.normal(size=300)
    a = float(fauzi_l_statistic(x)["estimate"])
    b = float(fauzi_l_statistic(x + 5.0)["estimate"])
    assert b - a == pytest.approx(5.0, abs=0.02)
