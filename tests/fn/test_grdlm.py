"""Tests for grdlm.geron_dataloader_minibatch."""

from morie.fn import _array_core as np

from morie.fn.grdlm import geron_dataloader_minibatch


def test_grdlm_basic():
    """Test basic functionality with an unshuffled epoch of size 100, batch size 10."""
    n = 100
    b = 10
    result = geron_dataloader_minibatch(n, b, shuffle=False, seed=42)

    # The function returns a RichResult whose dict interface exposes the payload keys.
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "statistic" not in result

    # Consecutive slices of [0..99] with batch size 10.
    expected_batches = [list(range(i, i + b)) for i in range(0, n, b)]
    expected_sizes = [b] * (n // b)

    assert result["batches"] == expected_batches
    assert result["batch_sizes"] == expected_sizes
    assert result["n_batches"] == n // b
    assert result["n"] == n
    assert result["permutation"] == list(range(n))
    assert result["covers_all"] is True
    assert result["drop_last"] is False
    assert result["estimate"] == expected_batches


def test_grdlm_edge():
    """Test edge cases: short tail kept vs dropped, and a shuffle that still covers all."""
    n = 5
    b = 2

    # Keep the short tail (default): one batch of size 2, one of size 2, one of size 1.
    kept = geron_dataloader_minibatch(n, b, shuffle=False)
    assert kept["batches"] == [[0, 1], [2, 3], [4]]
    assert kept["batch_sizes"] == [2, 2, 1]
    assert kept["n_batches"] == 3
    assert kept["covers_all"] is True
    # Documented: 3 batches covering indices 0..4.
    assert sorted(i for batch in kept["batches"] for i in batch) == list(range(n))

    # Drop the short tail: lose the final instance.
    dropped = geron_dataloader_minibatch(n, b, shuffle=False, drop_last=True)
    assert dropped["batches"] == [[0, 1], [2, 3]]
    assert dropped["batch_sizes"] == [2, 2]
    assert dropped["n_batches"] == 2
    assert dropped["drop_last"] is True
    # Independent count of indices that survive: 4 out of 5.
    assert sum(dropped["batch_sizes"]) == n - 1

    # Shuffled run: every index still appears exactly once (covers_all is True).
    shuffled = geron_dataloader_minibatch(6, 3, shuffle=True, seed=1)
    assert shuffled["n_batches"] == 2
    assert shuffled["batch_sizes"] == [3, 3]
    assert sorted(i for batch in shuffled["batches"] for i in batch) == list(range(6))
    assert shuffled["covers_all"] is True
    assert isinstance(shuffled["estimate"], list)
