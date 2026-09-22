"""Tests for gh_c11_11.ghosal_rescal_gp."""

import math

from morie.fn import _array_core as np

from morie.fn.gh_c11_11 import ghosal_rescal_gp


def test_gh_c11_11_basic():
    """Test basic functionality with documented defaults."""
    lengths = [2.0, 1.0, 0.25]
    h = 0.3
    result = ghosal_rescal_gp(lengths=lengths, h=h)

    assert "estimate" in result
    assert "correlation_by_length" in result
    assert "roughens_as_l_shrinks" in result
    assert "method" in result

    est = np.asarray(result["estimate"], dtype=float)
    assert np.all(np.isfinite(est))

    # estimate equals correlation at the smallest length
    expected_estimate = math.exp(-(h / lengths[-1]) ** 2)
    assert float(est) == expected_estimate

    # correlation_by_length matches the documented formula per-length
    expected_cors = [math.exp(-(h / l) ** 2) for l in lengths]
    got_cors = [float(v) for v in result["correlation_by_length"]]
    assert len(got_cors) == len(expected_cors)
    for got, exp in zip(got_cors, expected_cors):
        assert got == exp

    # roughness monotonicity holds for these lengths
    assert result["roughens_as_l_shrinks"] is True


def test_gh_c11_11_default_args():
    """Calling with no arguments uses the documented defaults."""
    result = ghosal_rescal_gp()
    expected = math.exp(-(0.3 / 0.25) ** 2)
    assert float(result["estimate"]) == expected
    assert len(result["correlation_by_length"]) == 3


def test_gh_c11_11_custom_lengths_and_h():
    """Custom lengths tuple and h produce the documented correlations."""
    lengths = (1.0, 0.5, 0.1)
    h = 0.4
    result = ghosal_rescal_gp(lengths=lengths, h=h)

    expected_cors = [math.exp(-(h / l) ** 2) for l in lengths]
    got_cors = [float(v) for v in result["correlation_by_length"]]
    for got, exp in zip(got_cors, expected_cors):
        assert got == exp

    # Shrinking l monotonically decreases the correlation
    for i in range(len(expected_cors) - 1):
        assert expected_cors[i + 1] <= expected_cors[i]
    assert result["roughens_as_l_shrinks"] is True

    # estimate is the correlation at the smallest length
    assert float(result["estimate"]) == expected_cors[-1]
