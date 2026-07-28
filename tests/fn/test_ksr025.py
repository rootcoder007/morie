"""Tests for ksr025 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr025 import kosorok_ch1_penalized_loglikelihood


def test_ksr025_basic():
    out = kosorok_ch1_penalized_loglikelihood(np.full(50, -1.0), J_eta=2.0,
                                              lambda_n=0.5)
    assert out["penalty"] == pytest.approx(0.25 * 4.0)  # lambda^2 J^2


def test_ksr025_edge():
    with pytest.raises(ValueError):
        kosorok_ch1_penalized_loglikelihood(np.zeros(10), J_eta=1.0, lambda_n=-1.0)
