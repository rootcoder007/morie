"""Tests for b204.burkov_lm_ch2_trigram_count."""

from morie.fn import _array_core as np

from morie.fn.b204 import burkov_lm_ch2_trigram_count


def test_b204_basic():
    """Test basic functionality with a known-seen trigram."""
    counts = {
        ("a", "b", "c"): 3,
        ("a", "b", "d"): 1,
        ("a", "b", "e"): 2,
        ("x", "y", "z"): 5,
    }
    # Context ("a", "b") has 3 + 1 + 2 = 6 occurrences.
    # Trigram ("a","b","c") has count 3.
    # MLE probability = 3 / 6 = 0.5.
    result = burkov_lm_ch2_trigram_count("c", "b", "a", counts=counts)
    assert isinstance(result, dict)
    assert "probability" in result
    assert "trigram_count" in result
    assert "context_count" in result
    assert "unseen" in result
    assert "smoothed" in result
    # Independent computation from the documented formula.
    expected_prob = 3.0 / 6.0
    assert result["probability"] == expected_prob
    assert result["trigram_count"] == 3
    assert result["context_count"] == 6
    assert result["unseen"] is False
    assert result["smoothed"] is False


def test_b204_edge():
    """Test edge cases: unseen trigram and add-alpha smoothing."""
    counts = {
        ("a", "b", "c"): 3,
        ("a", "b", "d"): 1,
        ("a", "b", "e"): 2,
    }
    # --- Unseen trigram: zero probability under MLE (den > 0). ---
    result_unseen = burkov_lm_ch2_trigram_count(
        "z", "b", "a", counts=counts
    )
    assert isinstance(result_unseen, dict)
    assert result_unseen["probability"] == 0.0
    assert result_unseen["trigram_count"] == 0
    assert result_unseen["context_count"] == 6
    assert result_unseen["unseen"] is True
    assert result_unseen["smoothed"] is False

    # --- Smoothing with add-alpha (Laplace, alpha=1), V=4. ---
    # numerator = 0 + 1, denominator = 6 + 1*4 = 10.
    result_smoothed = burkov_lm_ch2_trigram_count(
        "z", "b", "a", counts=counts, vocab_size=4, smoothing=1.0
    )
    assert isinstance(result_smoothed, dict)
    expected_smoothed = (0.0 + 1.0) / (6.0 + 1.0 * 4)
    assert result_smoothed["probability"] == expected_smoothed
    assert result_smoothed["smoothed"] is True
    assert result_smoothed["smoothing"] == 1.0
