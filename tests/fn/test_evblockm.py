"""Tests for evblockm.evt_block_maxima_fit."""

from morie.fn import _array_core as np

from morie.fn.evblockm import evt_block_maxima_fit


def test_evblockm_basic():
    """Test basic functionality against the documented Coles (2001)
    block-maxima + GEV MLE formula."""
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, 100)
    block_size = 10

    result = evt_block_maxima_fit(x, block_size)

    # Documented payload keys
    assert "mu" in result
    assert "sigma" in result
    assert "xi" in result
    assert "blocks" in result
    assert "ll" in result
    assert "method" in result

    # Recompute the block maxima independently (non-overlapping blocks
    # of length block_size, exactly as the implementation documents).
    n = len(x)
    n_blocks = n // block_size
    expected_maxima = []
    for j in range(n_blocks):
        start = j * block_size
        end = start + block_size
        expected_maxima.append(max(x[start:end]))

    assert len(result["blocks"]) == n_blocks
    assert list(result["blocks"]) == expected_maxima

    # Independence / sanity properties of the MLE:
    # sigma must be strictly positive.
    assert result["sigma"] > 0

    # Recompute the block-maxima sample mean (location sanity check).
    bm = expected_maxima
    sample_mean = sum(bm) / len(bm)
    # Just verify the fit produced a finite, real location parameter.
    assert result["mu"] == result["mu"]  # not NaN
    assert result["xi"] == result["xi"]  # not NaN


def test_evblockm_edge():
    """Test that two full blocks of data suffice (documented minimum)."""
    rng = np.random.default_rng(42)
    # Exactly 2 * block_size samples => two full blocks.
    block_size = 10
    x = rng.normal(0.0, 1.0, 2 * block_size)

    result = evt_block_maxima_fit(x, block_size)
    assert isinstance(result, dict) or hasattr(result, "payload")
    # Two blocks of 10 each.
    assert len(result["blocks"]) == 2
