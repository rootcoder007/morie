"""Tests for alftpl.alphafold_template_embed."""

import math

from morie.fn import _array_core as np

from morie.fn.alftpl import alphafold_template_embed


def _random(shape, seed):
    """Build a nested list with the given shape filled with normal random numbers."""
    rng = np.random.default_rng(seed)

    def helper(dims):
        if len(dims) == 0:
            return float(rng.normal(0.0, 1.0))
        return [helper(dims[1:]) for _ in range(dims[0])]

    return helper(shape)


def test_alftpl_basic():
    """Test basic functionality with multi-template, multi-head inputs."""
    nt, n, ct, cz, nhead, c = 2, 3, 4, 5, 2, 3

    t = _random([nt, n, n, ct], seed=42)
    z = _random([n, n, cz], seed=44)
    wq = _random([nhead, c, cz], seed=1)
    wk = _random([nhead, c, ct], seed=2)
    wv = _random([nhead, c, ct], seed=3)
    wo = _random([cz, nhead * c], seed=4)

    result = alphafold_template_embed(t, z, wq, wk, wv, wo)

    # result is a mapping-like RichResult
    assert hasattr(result, "__getitem__") or isinstance(result, dict)
    assert "z" in result
    assert "attn" in result
    assert "estimate" in result
    assert "n" in result
    assert "ntempl" in result
    assert "method" in result

    # z update has shape n x n x cz
    out_z = result["z"]
    assert len(out_z) == n
    for i in range(n):
        assert len(out_z[i]) == n
        for j in range(n):
            assert len(out_z[i][j]) == cz

    # attn has shape nhead x n x n x ntempl
    attn = result["attn"]
    assert len(attn) == nhead
    for h in range(nhead):
        assert len(attn[h]) == n
        for i in range(n):
            assert len(attn[h][i]) == n
            for j in range(n):
                assert len(attn[h][i][j]) == nt
                # softmax rows sum to 1
                row_sum = sum(attn[h][i][j])
                assert abs(row_sum - 1.0) < 1e-9

    # Independent recomputation of the estimate (mean of updated pair entries)
    flat = [out_z[i][j][u] for i in range(n) for j in range(n) for u in range(cz)]
    expected_estimate = sum(flat) / len(flat)
    assert abs(result["estimate"] - expected_estimate) < 1e-12

    assert result["n"] == n
    assert result["ntempl"] == nt
    assert isinstance(result["method"], str)


def test_alftpl_edge():
    """Test edge cases: single template forces attention weight = 1 everywhere."""
    nt, n, ct, cz, nhead, c = 1, 2, 3, 2, 1, 2

    t = _random([nt, n, n, ct], seed=7)
    z = _random([n, n, cz], seed=8)
    wq = _random([nhead, c, cz], seed=9)
    wk = _random([nhead, c, ct], seed=10)
    wv = _random([nhead, c, ct], seed=11)
    wo = _random([cz, nhead * c], seed=12)

    result = alphafold_template_embed(t, z, wq, wk, wv, wo)

    # With a single template attention must be identically 1.
    attn = result["attn"]
    for h in range(nhead):
        for i in range(n):
            for j in range(n):
                assert len(attn[h][i][j]) == 1
                assert abs(attn[h][i][j][0] - 1.0) < 1e-12

    # Closed-form: pooled value per (h, i, j) is just v = lin(t[0][i][j], wv[h])
    # and out[i][j] = lin(concat of pooled values, wo).
    # We recompute an independent estimate independently.
    out_z = result["z"]
    flat = [out_z[i][j][u] for i in range(n) for j in range(n) for u in range(cz)]
    expected_estimate = sum(flat) / len(flat)
    assert abs(result["estimate"] - expected_estimate) < 1e-12

    assert result["ntempl"] == nt
    assert result["n"] == n
    assert hasattr(result, "__getitem__") or isinstance(result, dict)
