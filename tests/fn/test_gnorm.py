"""Test group normalization."""

import pytest

from morie.fn import _array_core as np

from morie.fn.gnorm import gnorm


def test_gnorm_basic():
    """Test basic group norm."""
    x = np.random.randn(2, 32, 8, 8)
    result = gnorm(x, num_groups=8)
    assert result["output"].shape == x.shape


def test_gnorm_invalid_groups():
    """Test invalid groups."""
    x = np.random.randn(2, 32, 8, 8)
    try:
        gnorm(x, num_groups=7)
        assert False
    except ValueError:
        pass


def test_gnorm_shapes():
    """Test output shapes."""
    x = np.random.randn(4, 64, 16, 16)
    result = gnorm(x, num_groups=16)
    assert result["output"].shape == x.shape
    assert result["mean"].shape[:2] == (4, 16)


def test_gnorm_group_statistics():
    """Each group (all channels of the group, all positions) is centred and
    scaled: mean 0 and variance v / (v + eps) for the group variance v."""
    import math
    vals = [math.sin(0.37 * i) + 0.1 * (i % 7) for i in range(2 * 6 * 3 * 4)]
    x = np.array(vals).reshape(2, 6, 3, 4)
    r = gnorm(x, num_groups=3, epsilon=1e-6)
    out = r["output"].tolist()
    for b in range(2):
        for g in range(3):
            raw = [vals[((b * 6 + c) * 3 + h) * 4 + w] for c in (2 * g, 2 * g + 1)
                   for h in range(3) for w in range(4)]
            m = sum(raw) / 24
            v = sum((t - m) ** 2 for t in raw) / 24
            got = [out[b][c][h][w] for c in (2 * g, 2 * g + 1) for h in range(3) for w in range(4)]
            gm = sum(got) / 24
            assert abs(gm) < 1e-14
            assert sum((t - gm) ** 2 for t in got) / 24 == pytest.approx(v / (v + 1e-6), rel=1e-12)
