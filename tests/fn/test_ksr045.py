"""Tests for ksr045 (Kosorok shelf)."""

import numpy as np
import pytest

from morie.fn.ksr045 import kosorok_ch2_functional_delta_bootstrap


def test_ksr045_basic():
    rng = np.random.default_rng(13)
    X = rng.normal(1.0, 1.0, 300)
    boots = [np.array(float(rng.choice(X, 300, replace=True).mean())) for _ in range(200)]
    out = kosorok_ch2_functional_delta_bootstrap(lambda z: z**2, np.array(X.mean()),
                                                 boots, r_n=np.sqrt(300))
    assert abs(out["mean"]) < 1.0  # centred at the SAMPLE value


def test_ksr045_edge():
    with pytest.raises(ValueError):
        kosorok_ch2_functional_delta_bootstrap(lambda z: z, 1.0, [1.0], r_n=10.0)
