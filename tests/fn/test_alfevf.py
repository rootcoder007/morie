"""Tests for alfevf.alphafold_evoformer."""

from morie.fn import _array_core as np

from morie.fn.alfevf import alphafold_evoformer


def test_alfevf_basic():
    """Test basic functionality with nblock=0 (only the line-12 projection is used)."""
    rng = np.random.default_rng(44)
    s, n, cm, cz = 2, 3, 4, 5
    m = rng.normal(0, 1, (s, n, cm))
    z = rng.normal(0, 1, (n, n, cz))
    # With nblock=0 the block loop never runs, so only ``sout`` (line 12)
    # is consulted.  Per the docstring identity anchor, every projection
    # is the zero matrix, so the single representation comes out as zeros.
    w = {"sout": [[0.0] * cm for _ in range(cz)]}
    result = alphafold_evoformer(m, z, w, nblock=0)
    assert isinstance(result, dict)
    for key in ("m", "z", "s", "estimate", "method"):
        assert key in result
    # Identity stack: m and z are unchanged in shape
    assert len(result["m"]) == s
    assert len(result["m"][0]) == n
    assert len(result["m"][0][0]) == cm
    assert len(result["z"]) == n
    assert len(result["z"][0]) == n
    assert len(result["z"][0][0]) == cz
    # The single representation has one entry per residue
    assert len(result["s"]) == n


def test_alfevf_edge():
    """Test edge case with smallest dimensions."""
    rng = np.random.default_rng(45)
    s, n, cm, cz = 1, 2, 3, 4
    m = rng.normal(0, 1, (s, n, cm))
    z = rng.normal(0, 1, (n, n, cz))
    w = {"sout": [[0.0] * cm for _ in range(cz)]}
    result = alphafold_evoformer(m, z, w, nblock=0)
    assert isinstance(result, dict)
    for key in ("m", "z", "s", "estimate", "method"):
        assert key in result
    # Shape checks
    assert len(result["m"]) == s
    assert len(result["m"][0]) == n
    assert len(result["m"][0][0]) == cm
    assert len(result["z"]) == n
    assert len(result["z"][0]) == n
    assert len(result["z"][0][0]) == cz
    assert len(result["s"]) == n
