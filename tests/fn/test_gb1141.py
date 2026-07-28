"""Tests for gb1141 (Gibbons shelf)."""

import numpy as np
import pytest

from morie.fn.gb1141 import gibbons_tau_rho_relation


def test_gb1141_basic():
    assert gibbons_tau_rho_relation(0.4, 0.55)["consistent"] is True


def test_gb1141_edge():
    assert gibbons_tau_rho_relation(0.9, -0.9)["consistent"] is False
    with pytest.raises(ValueError):
        gibbons_tau_rho_relation(2.0, 0.0)
