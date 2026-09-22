"""Tests for ccdsgn.case_control."""

from morie.fn import _array_core as np

from morie.fn.ccdsgn import case_control


def test_ccdsgn_basic():
    """Test basic functionality with the documented 2x2 count interface."""
    rng = np.random.default_rng(42)
    n_cases = 100
    n_controls = 100
    p_case_exp = 0.4
    p_ctrl_exp = 0.25
    # Draw 0/1 indicators independently for cases and controls so that
    # the 2x2 cell counts are nonzero (the literature formula requires
    # all of a, b, c, d > 0 to avoid a zero cell).
    cases_ind = (rng.random(n_cases) < p_case_exp).astype(float)
    controls_ind = (rng.random(n_controls) < p_ctrl_exp).astype(float)

    a = float(cases_ind.sum())
    b = float(n_cases - cases_ind.sum())
    c = float(controls_ind.sum())
    d = float(n_controls - controls_ind.sum())

    result = case_control(cases_ind, controls_ind)

    assert isinstance(result, dict)
    # Function returns the documented keys.
    for key in ("estimate", "a", "b", "c", "d", "log_or", "se_log",
                "ci_low", "ci_high", "chisq", "significant", "n"):
        assert key in result

    # Independent recomputation of the literature formula.
    import math
    orr = (a * d) / (b * c)
    log_or = math.log(orr)
    se = math.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    n = a + b + c + d
    r1, r2 = a + b, c + d
    c1, c2 = a + c, b + d
    chi = n * (a * d - b * c) ** 2 / (r1 * r2 * c1 * c2)

    assert result["a"] == a
    assert result["b"] == b
    assert result["c"] == c
    assert result["d"] == d
    assert result["n"] == n
    assert abs(result["estimate"] - orr) < 1e-9
    assert abs(result["log_or"] - log_or) < 1e-9
    assert abs(result["se_log"] - se) < 1e-9
    assert abs(result["chisq"] - chi) < 1e-9


def test_ccdsgn_edge():
    """Test edge cases: a zero cell produces a finite OR via the (a,b)/(c,d)
    counts interface, while the CI/SE become NaN as documented by Woolf."""
    # (a, b, c, d) = (5, 0, 2, 3) -> b == 0 so the OR is inf, but a, c, d > 0.
    result = case_control((5, 0), (2, 3))

    assert isinstance(result, dict)
    assert result["a"] == 5
    assert result["b"] == 0
    assert result["c"] == 2
    assert result["d"] == 3
    # OR is inf because b == 0 while a*d > 0.
    assert result["estimate"] == float("inf")
    # A zero cell -> Woolf SE and CI are NaN per the source.
    import math
    assert math.isnan(result["se_log"])
    assert math.isnan(result["ci_low"])
    assert math.isnan(result["ci_high"])
