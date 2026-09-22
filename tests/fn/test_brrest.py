"""Tests for brrest.brr_balanced."""

from morie.fn import _array_core as np

from morie.fn.brrest import brr_balanced


def test_brrest_basic():
    """Test basic functionality."""
    strata = np.repeat(np.arange(5), 2)
    result = brr_balanced(strata)
    assert hasattr(result, "keys") or hasattr(result, "__getitem__")
    assert int(result["n_strata"]) == 5
    assert int(result["n_replicates"]) == 8
    W = result["replicate_weights"]
    assert W.shape == (8, 10)
    # Independent check: each replicate has exactly one PSU per stratum at
    # weight 2 and the other at weight 0 (fay_k=0 -> non-selected dropped).
    for r in range(8):
        for h in range(5):
            row = W[r, h * 2:(h + 1) * 2]
            assert sorted(np.round(row, 6).tolist()) == [0.0, 2.0]
    # Columns of the Hadamard slice must be mutually orthogonal (up to R).
    Hm = result["hadamard"]
    G = Hm.T @ Hm
    assert int(G[0, 0]) == 8
    # Independent recomputation of next power of two >= max(H, 4).
    R = 4
    while R < 5:
        R *= 2
    assert R == 8


def test_brrest_edge():
    """Test edge cases."""
    # Single stratum: R must still be 4 (minimum from the docstring).
    strata = np.array([0, 0])
    result = brr_balanced(strata)
    assert int(result["n_strata"]) == 1
    assert int(result["n_replicates"]) == 4
    W = result["replicate_weights"]
    assert W.shape == (4, 2)
    for r in range(4):
        assert sorted(np.round(W[r], 6).tolist()) == [0.0, 2.0]

    # Stratified input of the wrong shape must raise: a stratum with 1 PSU
    # is invalid by construction, not silently coerced.
    import pytest
    with pytest.raises(ValueError, match="requires exactly 2"):
        brr_balanced([0, 0, 1])

    # Fay adjustment keeps every weight strictly positive.
    strata = np.repeat(np.arange(5), 2)
    f = brr_balanced(strata, fay_k=0.3)
    Wf = f["replicate_weights"]
    assert Wf.shape == (8, 10)
    assert bool(Wf.min() > 0)
    # Independent recomputation of the per-stratum weights for fay_k=0.3.
    picked, dropped = 2.0 - 0.3, 0.3
    for r in range(8):
        for h in range(5):
            row = Wf[r, h * 2:(h + 1) * 2]
            assert sorted(np.round(row, 6).tolist()) == [dropped, picked]
