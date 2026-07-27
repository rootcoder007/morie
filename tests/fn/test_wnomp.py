"""Tests for wnomp."""

import numpy as np
import pytest

from morie.fn.wnomp import wnominate_probability


def test_wnomp_basic():
    assert wnominate_probability([0.0], [-1.0], [1.0])["p_yea"] == pytest.approx(0.5)
    assert wnominate_probability([-0.9], [-1.0], [1.0])["p_yea"] > 0.9


def test_wnomp_edge():
    lo = wnominate_probability([-0.3], [-1.0], [1.0], beta=1.0)["p_yea"]
    hi = wnominate_probability([-0.3], [-1.0], [1.0], beta=50.0)["p_yea"]
    assert hi > lo
    with pytest.raises(ValueError):
        wnominate_probability([0.0], [-1.0], [1.0], beta=-1.0)
