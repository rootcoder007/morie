"""Tests for convnx.convnext_block."""

from morie.fn import _array_core as np
from morie.fn.convnx import convnext_block


def test_convnx_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = convnext_block(x)
    assert isinstance(result, dict)
    assert "estimate" in result or "statistic" in result


def test_convnx_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = convnext_block(x)
    assert isinstance(result, dict)


def test_convnext_block_recomputed():
    """Depthwise k x k conv, LayerNorm, 1 -> expand -> 1 pointwise with GELU,
    residual scaled by gamma; the weights are replayed from the seed."""
    import math

    import pytest

    from morie.fn import _array_core as np

    M = [[1.0, 0.5, -0.2], [0.3, 2.0, 1.1], [-0.7, 0.4, 0.9]]
    k, e, gam = 3, 2, 0.5
    rng = np.random.default_rng(3)
    dw = [[float(rng.normal(0.0, 1.0)) / k for _ in range(k)] for _ in range(k)]
    w1 = [float(rng.normal(0.0, 0.02)) for _ in range(e)]
    w2 = [float(rng.normal(0.0, 0.02)) for _ in range(e)]
    conv = [
        [
            sum(
                M[min(max(i + a, 0), 2)][min(max(j + b, 0), 2)] * dw[a + 1][b + 1]
                for a in (-1, 0, 1)
                for b in (-1, 0, 1)
            )
            for j in range(3)
        ]
        for i in range(3)
    ]
    flat = [v for r in conv for v in r]
    mu = sum(flat) / 9
    var = sum((v - mu) ** 2 for v in flat) / 9
    gelu = lambda z: 0.5 * z * (1 + math.erf(z / math.sqrt(2)))  # noqa: E731
    out = [
        [
            M[i][j] + gam * sum(gelu((conv[i][j] - mu) / math.sqrt(var + 1e-6) * w1[q]) * w2[q] for q in range(e))
            for j in range(3)
        ]
        for i in range(3)
    ]
    r = convnext_block(M, kernel=3, expand=e, layer_scale=gam, seed=3)
    for i in range(3):
        assert r["out"][i] == pytest.approx(out[i], rel=1e-13)
    assert convnext_block(M, kernel=3)["residual_norm"] == 0.0
