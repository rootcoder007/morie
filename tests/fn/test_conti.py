"""Tests for conti.continued_fraction_pi."""

from morie.fn.conti import continued_fraction_pi


def test_conti_basic():
    """Test basic functionality with the documented maximum n=15."""
    n = 15
    result = continued_fraction_pi(n)
    assert isinstance(result, dict)
    # The function returns a RichResult dict with these documented keys.
    for key in ("estimate", "terms", "convergents",
                "numerator", "denominator", "error", "n", "method"):
        assert key in result

    # n is echoed back as an int.
    assert result["n"] == 15

    # We get 15 partial quotients and 15 convergents.
    assert len(result["terms"]) == 15
    assert len(result["convergents"]) == 15

    # The (numerator, denominator) pair is the last convergent.
    last_h, last_k = result["convergents"][-1]
    assert result["numerator"] == last_h
    assert result["denominator"] == last_k

    # estimate equals numerator / denominator (computed independently).
    assert result["estimate"] == last_h / float(last_k)

    # error is documented as estimate - pi; compute pi literally in the test.
    pi_lit = 3.141592653589793
    expected_error = result["estimate"] - pi_lit
    assert result["error"] == expected_error

    # The classical convergents mentioned in the docstring: 3/1 and 355/113.
    # 355/113 is the 4th convergent (n=4 in 1-indexed partial-quotient counting).
    assert result["convergents"][0] == (3, 1)
    assert result["convergents"][3] == (355, 113)

    # 355/113 is accurate to ~7 figures, so its error magnitude < 1e-6.
    err355 = 355 / 113.0 - pi_lit
    assert abs(err355) < 1e-6


def test_conti_edge():
    """Test edge cases: n=1 (smallest valid input)."""
    n = 1
    result = continued_fraction_pi(n)
    assert isinstance(result, dict)
    assert result["n"] == 1
    # First convergent of pi's continued fraction is 3/1.
    assert result["convergents"] == [(3, 1)]
    assert result["numerator"] == 3
    assert result["denominator"] == 1
    # estimate = numerator / denominator, computed independently.
    assert result["estimate"] == 3 / 1.0
