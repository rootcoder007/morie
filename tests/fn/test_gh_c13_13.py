"""Tests for gh_c13_13.ghosal_cox_model."""

from morie.fn import _array_core as np

from morie.fn.gh_c13_13 import ghosal_cox_model


def test_gh_c13_13_basic():
    """Test basic functionality."""
    beta = 0.7
    z = (0.0, 1.0)
    t = 1.0
    c = 2.0

    result = ghosal_cox_model(beta=beta, z=z, t=t, c=c)

    # Key documented in the docstring / RichResult payload.
    assert "estimate" in result

    # The estimate should be a finite scalar.
    estimate = result["estimate"]
    assert np.all(np.isfinite(np.asarray(estimate, dtype=float)))

    # Independent expectation from the documented formula:
    # H_i = t * exp(beta * z_i); estimate = H[1] / H[0].
    H0 = t * np.exp(beta * z[0])
    H1 = t * np.exp(beta * z[1])
    expected_ratio = float(H1) / float(H0)
    assert np.isclose(float(estimate), expected_ratio)

    # Additional documented keys.
    assert "cum_hazards" in result
    cum_hazards = result["cum_hazards"]
    assert len(cum_hazards) == 2
    assert np.isclose(float(cum_hazards[0]), float(H0))
    assert np.isclose(float(cum_hazards[1]), float(H1))

    assert result["proportional"] is True
    assert isinstance(result["method"], str)


def test_gh_c13_13_edge():
    """Test edge cases."""
    result = ghosal_cox_model(beta=0.7, z=(0.0, 1.0), t=1.0, c=2.0)

    # Documented return keys; "n" is not one of them.
    assert "estimate" in result
    assert "cum_hazards" in result
    assert "proportional" in result
    assert "method" in result
