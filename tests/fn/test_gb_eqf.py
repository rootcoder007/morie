"""Tests for gb_eqf (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb_eqf import gibbons_emp_quantile


def test_gb_eqf_basic():
    data = [3.0, 1.0, 4.0, 1.5, 5.0]
    assert gibbons_emp_quantile(0.2, data)["quantile"] == 1.0
    assert gibbons_emp_quantile(1.0, data)["quantile"] == 5.0


def test_gb_eqf_edge():
    with pytest.raises(ValueError):
        gibbons_emp_quantile(0.0, [1.0, 2.0])
