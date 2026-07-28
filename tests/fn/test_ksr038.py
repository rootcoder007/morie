"""Tests for ksr038 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr038 import kosorok_ch2_donsker_uniform_entropy


def test_ksr038_basic():
    out = kosorok_ch2_donsker_uniform_entropy(lambda e: (1 / e) ** 2, 3.0)
    assert out["conditions_met"] is True


def test_ksr038_edge():
    # Donsker keys on P*F^2, not P*F
    bad = kosorok_ch2_donsker_uniform_entropy(lambda e: (1 / e) ** 2, np.inf)
    assert bad["envelope_sq_integrable"] is False
