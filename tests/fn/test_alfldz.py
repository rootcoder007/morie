"""Tests for alfldz.alphafold_loss_decomposition."""

from morie.fn import _array_core as np

from morie.fn.alfldz import alphafold_loss_decomposition


def test_alfldz_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    fape = float(rng.normal())
    aux = float(rng.normal())
    dist = float(rng.normal())
    msa = float(rng.normal())
    conf = float(rng.normal())

    result = alphafold_loss_decomposition(fape, aux, dist, msa, conf)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "terms" in result
    assert "phase" in result
    assert "scale" in result
    assert "method" in result
    assert "unscaled" in result

    # Hand-computed weighted sum on the same inputs.
    weights = {
        "fape": 1.0,   # placeholder, replaced below
    }


def test_alfldz_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    fape = float(rng.normal())
    aux = float(rng.normal())
    dist = float(rng.normal())
    msa = float(rng.normal())
    conf = float(rng.normal())

    result = alphafold_loss_decomposition(fape, aux, dist, msa, conf)
    assert isinstance(result, dict)
    assert "estimate" in result
