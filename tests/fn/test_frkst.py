"""Tests for frkst.fork_structure."""

import numpy as np
import pytest

from morie.fn.frkst import fork_structure


def test_frkst_basic():
    rng = np.random.default_rng(42)
    b = rng.normal(size=3000)
    a = b + rng.normal(scale=0.7, size=3000)
    c = b + rng.normal(scale=0.7, size=3000)
    out = fork_structure(a, b, c)
    assert out["consistent_with_fork"] is True
    assert out["marginally_dependent"] is True
    assert out["conditionally_independent"] is True


def test_frkst_edge():
    with pytest.raises(ValueError):
        fork_structure([1.0, 2.0], [1.0], [1.0, 2.0])  # length mismatch
