"""Tests for fzb1b5.fauzi_assumptions_b1_b5."""

from morie.fn import _array_core as np

from morie.fn.fzb1b5 import fauzi_assumptions_b1_b5


def test_fzb1b5_basic():
    """Test basic functionality with the default Gaussian kernel."""
    result = fauzi_assumptions_b1_b5(h=0.1, n=100)
    # Documented keys returned by the function.
    for key in ("b1", "b2", "b3", "b4", "b5", "mu4", "mass", "method"):
        assert key in result
    # Gaussian is non-negative, symmetric about 0, integrates to 1,
    # and has finite fourth moment, so B1 and B2 must be True.
    assert result["b1"] is True
    assert result["b2"] is True
    # Finite-sample form of B3: 0 < h < 1 and n*h > 1.
    assert result["b3"] is True
    # B4 is unchecked by the function; with no `smooth` argument it is None.
    assert result["b4"] is None
    assert result["b5"] is None
    # With the default Gaussian kernel the mass should be ~1 and mu4 = 3.
    assert np.all(np.isfinite(np.asarray(result["mass"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(result["mu4"], dtype=float)))
    expected_mass = 1.0
    expected_mu4 = 3.0
    assert abs(float(result["mass"]) - expected_mass) < 1e-3
    assert abs(float(result["mu4"]) - expected_mu4) < 1e-2
    assert result["method"] == "assumptions B1-B5 of the bias-reduced KDFE"


def test_fzb1b5_edge():
    """Test edge cases for the documented argument shapes."""
    # Scalar array input for `h` and `n` still triggers the documented
    # finite-sample B3 check; passing h=5 (>= 1) must violate B3.
    result = fauzi_assumptions_b1_b5(h=np.array(5.0), n=np.array(10))
    assert result["b3"] is False
    # If either h or n is omitted, B3 is reported as None.
    result_no_n = fauzi_assumptions_b1_b5(h=0.2)
    assert result_no_n["b3"] is None
    # Passing a callable kernel uses it instead of the default Gaussian.
    result_callable = fauzi_assumptions_b1_b5(
        kernel=lambda w: float(np.exp(-0.5 * w * w) / np.sqrt(2.0 * np.pi)),
        h=0.1,
        n=100,
    )
    assert result_callable["b1"] is True
    assert result_callable["b2"] is True
