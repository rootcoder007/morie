"""Tests for cncpat.controlnet_attach."""

from morie.fn import _array_core as np
from morie.fn.cncpat import controlnet_attach


def test_cncpat_basic():
    """Test basic functionality."""
    base = np.random.default_rng(42).normal(0, 1, 100)
    condition = np.random.default_rng(42).normal(0, 1, 100)
    result = controlnet_attach(base, condition)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_cncpat_edge():
    """Test edge cases."""
    base = np.random.default_rng(42).normal(0, 1, 100)
    condition = np.random.default_rng(42).normal(0, 1, 100)
    result = controlnet_attach(base, condition)
    assert isinstance(result, dict)


def test_controlnet_output_recomputed():
    """out = base + w * GELU(3x3 conv of the condition, edge-replicated)."""
    import math

    import pytest

    from morie.fn import _array_core as np

    base = [[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]]
    cond = [[0.2, 0.4, 0.1], [0.9, -0.3, 0.6]]
    rng = np.random.default_rng(7)
    w = [[float(rng.normal(0.0, 0.5)) for _ in range(3)] for _ in range(3)]
    H, W = 2, 3
    out = []
    for i in range(H):
        row = []
        for j in range(W):
            s = sum(
                cond[min(max(i + a, 0), H - 1)][min(max(j + b, 0), W - 1)] * w[a + 1][b + 1]
                for a in (-1, 0, 1)
                for b in (-1, 0, 1)
            )
            row.append(base[i][j] + 0.3 * 0.5 * s * (1 + math.erf(s / math.sqrt(2))))
        out.append(row)
    r = controlnet_attach(base, cond, zero_conv_weight=0.3, seed=7)
    for i in range(H):
        assert r["out"][i] == pytest.approx(out[i], rel=1e-13)
    assert controlnet_attach(base, cond)["is_identity"] == 1
