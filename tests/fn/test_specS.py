"""Tests for specS.speculative_decoding (Leviathan, Kalman & Matias 2023)."""

import pytest

from morie.fn.specS import speculative_decoding


def test_specS_basic():
    """alpha = sum min(p, q) = 1 - TV; E[tokens] = (1 - alpha^(g+1))/(1 - alpha),
    which is also the truncated geometric sum 1 + alpha + ... + alpha^g."""
    q = [0.5, 0.3, 0.2]
    p = [0.4, 0.4, 0.2]
    r = speculative_decoding(q, p, gamma=3)
    alpha = 0.4 + 0.3 + 0.2
    assert r["alpha"] == pytest.approx(alpha, abs=1e-15)
    assert r["tv_distance"] == pytest.approx(0.5 * sum(abs(a - b) for a, b in zip(p, q)), abs=1e-15)
    assert r["expected_tokens"] == pytest.approx(sum(alpha ** k for k in range(4)), rel=1e-14)
    assert r["max_tokens"] == 4.0


def test_specS_edge():
    """Identical distributions accept everything (gamma + 1 tokens);
    disjoint support accepts nothing (1 token); bad inputs raise."""
    assert speculative_decoding([0.3, 0.7], [0.3, 0.7], gamma=5)["expected_tokens"] == 6.0
    assert speculative_decoding([1.0, 0.0], [0.0, 1.0], gamma=5)["expected_tokens"] == 1.0
    with pytest.raises(ValueError):
        speculative_decoding([1.2, -0.2], [0.5, 0.5])
    with pytest.raises(ValueError):
        speculative_decoding([0.6, 0.6], [0.5, 0.5])
    with pytest.raises(ValueError):
        speculative_decoding([0.5, 0.5], [0.5, 0.5], gamma=0)
    with pytest.raises(ValueError):
        speculative_decoding([1.0], [1.0])
