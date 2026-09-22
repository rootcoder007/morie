"""Tests for bndest.bound_estimation."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.bndest import bound_estimation


def test_bndest_basic():
    """Test basic functionality: missing-outcome bounds on a mean."""
    rng = np.random.default_rng(43)
    y = rng.normal(0, 1, 100)
    observed = rng.uniform(0, 1, 100) < 0.7
    # The function's third positional argument is `support`, a (K0, K1)
    # tuple -- an assumption about the outcome's logical range, NOT a
    # vector of moments. Pass a valid support for these data.
    support = (-5.0, 5.0)

    result = bound_estimation(y, observed, support)

    # The result is a RichResult; for this code base it behaves like a
    # dict and exposes the documented keys.
    assert isinstance(result, dict)

    # Documented keys for the no-treatment branch.
    for key in ("lower", "upper", "width", "p_observed",
                "identified", "n", "method"):
        assert key in result, f"missing key {key!r} in result"

    # Compute the expected quantities independently from the documented
    # formula: bounds on E[Y] are [E[Y|obs]*P(obs) + K0*(1-P(obs)),
    #                                 E[Y|obs]*P(obs) + K1*(1-P(obs))].
    yv = np.asarray(y, dtype=float).ravel()
    obs = np.asarray(observed, dtype=bool).ravel()
    K0, K1 = float(support[0]), float(support[1])
    p = float(obs.mean())
    m = float(yv[obs].mean())
    exp_lower = m * p + K0 * (1.0 - p)
    exp_upper = m * p + K1 * (1.0 - p)
    exp_width = (K1 - K0) * (1.0 - p)

    assert result["lower"] == exp_lower
    assert result["upper"] == exp_upper
    assert result["width"] == exp_width

    # p_observed is the fraction of units whose outcome was seen.
    assert result["p_observed"] == p
    assert result["n"] == yv.size

    # Width identity stated in the docstring: width == (K1-K0)*(1-P(obs)).
    assert result["width"] == (K1 - K0) * (1.0 - result["p_observed"])

    # Bounds are ordered (lower <= upper) and contained within support.
    assert result["lower"] <= result["upper"]
    assert K0 <= result["lower"]
    assert result["upper"] <= K1

    # Nothing is missing only when p == 1, in which case the bounds
    # collapse to the sample mean and the parameter is "identified".
    if p == 1.0:
        assert result["identified"] is True
        assert result["lower"] == result["upper"] == m
    else:
        assert result["identified"] is False


def test_bndest_edge():
    """Test edge cases: counterfactual bounds with a treatment vector."""
    rng = np.random.default_rng(43)
    n = 100
    y = rng.normal(0, 1, n)
    # Binary treatment indicator (0/1). `observed` is ignored when
    # `treatment` is supplied -- the missingness becomes the
    # counterfactual one (treated units' Y(0) are missing and vice
    # versa), exactly as the docstring describes.
    treatment = (rng.uniform(0, 1, n) < 0.5).astype(int)
    support = (-5.0, 5.0)

    result = bound_estimation(y, None, support, treatment=treatment)

    assert isinstance(result, dict)

    # Documented keys for the treatment branch.
    for key in ("ate_lower", "ate_upper", "ate_width",
                "y1_bounds", "y0_bounds", "p_treated",
                "contains_zero", "n", "method"):
        assert key in result, f"missing key {key!r} in result"

    # Independent computation of the ATE bounds from the documented
    # construction: bounds on E[Y(1)] use only the treated units' Y;
    # bounds on E[Y(0)] use only the untreated units' Y. The ATE bounds
    # difference them as stated in the docstring.
    Tv = np.asarray(treatment).ravel().astype(bool)
    yv = np.asarray(y, dtype=float).ravel()
    K0, K1 = float(support[0]), float(support[1])

    def one_mean(seen):
        p = float(seen.mean())
        m = float(yv[seen].mean()) if p > 0 else 0.0
        return m * p + K0 * (1 - p), m * p + K1 * (1 - p), p

    lo1, hi1, p1 = one_mean(Tv)
    lo0, hi0, p0 = one_mean(~Tv)
    exp_ate_lo = lo1 - hi0
    exp_ate_hi = hi1 - lo0

    assert result["ate_lower"] == exp_ate_lo
    assert result["ate_upper"] == exp_ate_hi
    assert result["ate_width"] == exp_ate_hi - exp_ate_lo

    # The docstring guarantees: ATE bounds have width exactly K1 - K0
    # and ALWAYS contain zero. This is the central no-assumption
    # property of Manski (1990) -- an implementation whose bounds
    # exclude zero would have smuggled in an extra assumption.
    assert result["ate_width"] == K1 - K0
    assert result["contains_zero"] is True
    assert result["ate_lower"] <= 0.0 <= result["ate_upper"]

    # The two potential-outcome bound pairs are exposed as 2-tuples.
    assert result["y1_bounds"] == (lo1, hi1)
    assert result["y0_bounds"] == (lo0, hi0)
    assert result["p_treated"] == p1
    assert result["n"] == n
