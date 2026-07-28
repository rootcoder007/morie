"""Tests for gb821t (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb821t import gibbons_wrs_ties


def test_gb821t_basic():
    rng = np.random.default_rng(7)
    x = np.round(rng.standard_normal(20), 0); y = np.round(rng.standard_normal(22), 0)
    from morie.fn.gb661t import gibbons_mw_ties
    assert gibbons_wrs_ties(x, y)["var_corrected"] == pytest.approx(
        gibbons_mw_ties(x, y)["var_corrected"])


def test_gb821t_edge():
    with pytest.raises(ValueError):
        gibbons_wrs_ties([], [1.0])
