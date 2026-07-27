"""Tests for nonresp.nonresponse_adjustment."""

import numpy as np
import pytest

from morie.fn.nonresp import nonresponse_adjustment


def test_nonresp_basic():
    # w_adj = (2, 1); Hajek mean = (2*1 + 1*3)/3
    result = nonresponse_adjustment([1.0, 3.0], [1.0, 1.0], [0.5, 1.0])
    assert result["weights_adjusted"] == pytest.approx([2.0, 1.0])
    assert result["estimate"] == pytest.approx(5.0 / 3.0)


def test_nonresp_edge():
    with pytest.raises(ValueError):
        nonresponse_adjustment([1.0], [1.0], [0.0])  # phi = 0
    with pytest.raises(ValueError):
        nonresponse_adjustment([1.0], [-1.0], [0.5])  # negative weight
