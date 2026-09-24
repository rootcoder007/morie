"""Tests for km143.kamath_ch9_fom_loss."""

import math

from morie.fn import _array_core as np

from morie.fn.km143 import kamath_ch9_fom_loss


def test_km143_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n_frames = 5
    n_timestamps = 8
    raw = [[float(rng.uniform(0, 1)) for _ in range(n_timestamps)]
           for _ in range(n_frames)]
    P = [[v / sum(row) for v in row] for row in raw]

    r_i = list(range(n_frames))
    t_i = list(range(n_frames))
    result = kamath_ch9_fom_loss(r_i, t_i, n_frames, P)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    assert result["n_reordered"] == n_frames
    assert result["n"] == n_frames
    assert len(result["per_frame"]) == n_frames
    assert all(math.isfinite(v) for v in result["per_frame"])


def test_km143_edge():
    """Test edge case using the smallest valid PMF from the docstring."""
    P = [[0.5, 0.5], [0.25, 0.75]]
    r_i = [0, 1]
    t_i = [0, 1]
    result = kamath_ch9_fom_loss(r_i, t_i, 2, P)
    assert isinstance(result, dict)
    assert math.isfinite(result["estimate"])
    expected = math.log(2) - math.log(0.75)
    assert abs(result["estimate"] - expected) < 1e-12
    assert result["n_reordered"] == 2
    assert len(result["per_frame"]) == 2
