"""Tests for eslsmp.esl_subsampling."""

from morie.fn import _array_core as np

from morie.fn.eslsmp import esl_subsampling


def test_eslsmp_basic():
    """Test basic functionality."""
    eta = float(np.random.default_rng(42).normal(0, 1, 1)[0])
    result = esl_subsampling(eta)
    assert isinstance(result, dict)
    # Documented keys per the docstring.
    for key in ("estimate", "eta", "n", "n_sampled", "indices",
                "cost_multiplier", "method"):
        assert key in result
    # With n=None, estimate collapses to eta itself.
    assert result["estimate"] == eta
    assert result["eta"] == eta
    assert result["n"] is None
    assert result["n_sampled"] is None
    assert result["indices"] is None
    assert result["cost_multiplier"] == eta


def test_eslsmp_edge():
    """Test edge cases."""
    # eta == 1.0 is the upper bound of the documented interval (0, 1].
    eta = 1.0
    n = 7
    result = esl_subsampling(eta, n=n)
    assert isinstance(result, dict)
    # Independent expectation from the documented formula: n_sampled = round(eta * n).
    expected_n_sampled = max(1, int(round(eta * n)))
    assert result["n_sampled"] == expected_n_sampled
    assert result["estimate"] == expected_n_sampled
    assert result["eta"] == eta
    assert result["n"] == n
    assert result["cost_multiplier"] == eta
    # Sampled without replacement: indices are unique and sorted, in [0, n).
    idx = result["indices"]
    assert idx is not None
    assert len(idx) == expected_n_sampled
    assert list(idx) == sorted(idx)
    assert len(set(idx)) == expected_n_sampled
    assert all(0 <= int(i) < n for i in idx)
