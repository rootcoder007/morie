"""Tests for gcnII.gcnii."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.gcnII import gcnii


def test_gcnII_basic():
    """Test basic functionality of the GCNII recursion with W = I."""
    # Use lists-of-lists for the adjacency matrix to be safe with the shim
    A = [
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ]
    H0 = [
        [1.0, 0.5],
        [0.5, 1.0],
        [1.0, 1.0],
        [0.0, 0.5],
    ]
    alpha = 0.1
    beta = 0.5
    K = 4
    result = gcnii(A, H0, alpha, beta, K)

    # Function returns a RichResult with payload; assert documented keys
    payload = result.payload
    assert "estimate" in payload
    assert "H" in payload
    assert "alpha" in payload
    assert "beta" in payload
    assert "K" in payload
    assert "n" in payload
    assert "method" in payload

    # Metadata should reflect what we passed in
    assert payload["alpha"] == alpha
    assert payload["beta"] == beta
    assert payload["K"] == K
    assert payload["n"] == len(A)

    # H should have shape (n, H0_width) = (4, 2)
    H_out = payload["H"]
    assert len(H_out) == 4
    assert all(len(row) == 2 for row in H_out)

    # ReLU output must be non-negative
    flat = [v for row in H_out for v in row]
    assert all(v >= 0.0 for v in flat)

    # Independent recomputation of `estimate`
    assert payload["estimate"] == sum(flat) / len(flat)


def test_gcnII_higher_layers():
    """Test that a different K changes the recursion depth, with K=1 a single step."""
    A = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
    H0 = [
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
    ]

    r1 = gcnii(A, H0, alpha=0.1, beta=0.5, K=1).payload
    r3 = gcnii(A, H0, alpha=0.1, beta=0.5, K=3).payload

    # Both should report the requested K
    assert r1["K"] == 1
    assert r3["K"] == 3

    # After more layers the state should generally differ
    assert r1["H"] != r3["H"]
