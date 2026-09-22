"""Tests for alfcnf.alphafold_confidence."""

from morie.fn import _array_core as np

from morie.fn.alfcnf import alphafold_confidence


def test_alfcnf_basic():
    """Test basic functionality.

    With the final projection w3 set to zeros, the bin logits are flat, so
    the softmax distribution is uniform.  The reported pLDDT is then the
    arithmetic mean of the bin centres, which is the first closed-form
    anchor documented for this function.
    """
    n, cs = 4, 3
    rng = np.random.default_rng(42)
    s = rng.normal(0, 1, (n, cs))
    # Dimensions: s -> w1 -> w2 -> w3(nbins)
    c_hidden = 5
    nbins = 50
    w1 = rng.normal(0, 1, (cs, c_hidden))
    w2 = rng.normal(0, 1, (c_hidden, c_hidden))
    # Flat logits -> uniform softmax over bins.
    w3 = np.zeros((nbins, c_hidden))

    result = alphafold_confidence(s, w1, w2, w3)
    assert isinstance(result, dict)
    assert "estimate" in result
    # The expected mean of the default bin centres [1, 3, ..., 99].
    expected_mean = sum(range(1, 100, 2)) / 50.0
    assert result["estimate"] == expected_mean
    assert result["n"] == n
    assert len(result["plddt"]) == n
    # Uniform distribution: each bin gets probability 1/nbins.
    expected_p = [1.0 / nbins] * nbins
    for row in result["p"]:
        assert all(abs(row[b] - expected_p[b]) < 1e-12 for b in range(nbins))


def test_alfcnf_edge():
    """Test edge cases: single residue, and no rtrue -> loss is None."""
    n, cs = 1, 2
    rng = np.random.default_rng(42)
    s = rng.normal(0, 1, (n, cs))
    c_hidden = 3
    nbins = 50
    w1 = rng.normal(0, 1, (cs, c_hidden))
    w2 = rng.normal(0, 1, (c_hidden, c_hidden))
    w3 = np.zeros((nbins, c_hidden))

    result = alphafold_confidence(s, w1, w2, w3)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert result["loss"] is None
    assert result["n"] == n
    assert len(result["plddt"]) == n
