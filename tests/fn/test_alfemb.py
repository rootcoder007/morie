"""Tests for alfemb.alphafold_embedding_init."""

from morie.fn import _array_core as np

from morie.fn.alfemb import alphafold_embedding_init


def test_alfemb_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)

    # Dimensions
    n = 4   # number of residues
    ctf = 3 # target feature dim
    s = 2   # number of MSA sequences
    cmf = 3 # MSA feature dim
    cz = 2  # pair representation dim
    cm = 2  # MSA representation dim

    # Build inputs as nested Python lists (since function uses list comprehensions)
    target_feat = rng.normal(0, 1, (n, ctf)).tolist()
    residue_index = [float(i) for i in range(n)]
    msa_feat = rng.normal(0, 1, (s, n, cmf)).tolist()
    wa = rng.normal(0, 1, (cz, ctf)).tolist()
    wb = rng.normal(0, 1, (cz, ctf)).tolist()
    # bins default has 65 entries ([-32, ..., 32])
    n_bins = 65
    wrel = rng.normal(0, 1, (cz, n_bins)).tolist()
    wmsa = rng.normal(0, 1, (cm, cmf)).tolist()
    wtgt = rng.normal(0, 1, (cm, ctf)).tolist()

    result = alphafold_embedding_init(
        target_feat, residue_index, msa_feat,
        wa, wb, wrel, wmsa, wtgt
    )

    # Result is a RichResult; support both dict-like and attribute access
    # Check key presence
    assert "estimate" in result
    assert "n" in result
    assert "method" in result
    assert "z" in result
    assert "m" in result
    assert "pos" in result

    # Check n
    assert result["n"] == n

    # Check z shape: n x n x cz
    assert len(result["z"]) == n
    assert all(len(row) == n for row in result["z"])
    assert all(len(row[i]) == cz for row in result["z"] for i in range(n))

    # Check m shape: s x n x cm
    assert len(result["m"]) == s
    assert all(len(row) == n for row in result["m"])
    assert all(len(row[i]) == cm for row in result["m"] for i in range(n))

    # Check pos shape: n x n x cz
    assert len(result["pos"]) == n
    assert all(len(row) == n for row in result["pos"])
    assert all(len(row[i]) == cz for row in result["pos"] for i in range(n))


def test_alfemb_edge():
    """Test edge cases: n=1 (single residue)."""
    rng = np.random.default_rng(42)

    n = 1
    ctf = 3
    s = 1
    cmf = 3
    cz = 2
    cm = 2

    target_feat = rng.normal(0, 1, (n, ctf)).tolist()
    residue_index = [0.0]
    msa_feat = rng.normal(0, 1, (s, n, cmf)).tolist()
    wa = rng.normal(0, 1, (cz, ctf)).tolist()
    wb = rng.normal(0, 1, (cz, ctf)).tolist()
    n_bins = 65
    wrel = rng.normal(0, 1, (cz, n_bins)).tolist()
    wmsa = rng.normal(0, 1, (cm, cmf)).tolist()
    wtgt = rng.normal(0, 1, (cm, ctf)).tolist()

    result = alphafold_embedding_init(
        target_feat, residue_index, msa_feat,
        wa, wb, wrel, wmsa, wtgt
    )

    assert "estimate" in result
    assert result["n"] == n
    assert len(result["z"]) == n
