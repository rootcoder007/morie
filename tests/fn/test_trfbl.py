"""Tests for trfbl.transformer_block."""

from morie.fn import _array_core as np
import pytest

from morie.fn.trfbl import transformer_block


def test_trfbl_output_shape_and_layernorm_statistics():
    """Post-LN block: the output IS a LayerNorm output, so every row has
    mean 0 and variance 1 -- exact, not approximate."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 8))
    r = transformer_block(x, num_heads=2, seed=0)
    out = np.asarray(r["output"], dtype=float)
    assert out.shape == (6, 8)
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-8)
    # LayerNorm's epsilon leaves the variance ~1e-5 shy of exactly 1.
    np.testing.assert_allclose(out.var(axis=1), 1.0, atol=1e-4)


def test_trfbl_is_reproducible_and_seed_sensitive():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 8))
    a = np.asarray(transformer_block(x, num_heads=2, seed=3)["output"], dtype=float)
    b = np.asarray(transformer_block(x, num_heads=2, seed=3)["output"], dtype=float)
    c = np.asarray(transformer_block(x, num_heads=2, seed=4)["output"], dtype=float)
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert not np.allclose(a, c, atol=1e-6)  # different weights, different output


def test_trfbl_intermediate_h1_is_also_normalised():
    rng = np.random.default_rng(2)
    r = transformer_block(rng.normal(size=(5, 8)), num_heads=2)
    h1 = np.asarray(r["h1"], dtype=float)
    np.testing.assert_allclose(h1.mean(axis=1), 0.0, atol=1e-8)


def test_trfbl_rejects_indivisible_heads():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        transformer_block(rng.normal(size=(4, 7)), num_heads=2)
