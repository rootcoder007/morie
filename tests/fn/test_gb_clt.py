"""Tests for gb_clt (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb_clt import gibbons_clt


def test_gb_clt_basic():
    assert gibbons_clt(xbar=0.2, n=100)["z"] == pytest.approx(2.0)


def test_gb_clt_edge():
    with pytest.raises(ValueError):
        gibbons_clt(xbar=0.2)  # n missing
