"""Tests for irtgr — graded response model."""

from morie.fn.irtgr import irtgr


def test_irtgr_basic(mapq_df):
    items = [c for c in mapq_df.columns if c.startswith(("EE", "EA", "UA", "ER")) and c[-1].isdigit()]
    result = irtgr(mapq_df[items].values)
    assert hasattr(result, "item_params")


def test_cheatsheet():
    from morie.fn.irtgr import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0


# ---------------------------------------------------------------------------
# Second identity test -- Samejima (1969), Psychometric Monograph No. 17.
#
# The worked-example route is not open here: Samejima's Ch 5 tables are
# goodness-of-fit frequency distributions for the LIS scale (Table 5-2), not
# a G-matrix-style "feed these numbers in, get these numbers out" example we
# could transcribe. What the chapter DOES give is closed-form properties the
# operating characteristic must satisfy, which is exactly what an identity
# test is for.
#
# Ch 5 p.23-24 defines, for the graded response on item g:
#   (5-3)  b_(x+1) > b_x                    thresholds are strictly ordered
#   (5-4)  P_x(theta) = P*_x(theta) - P*_(x+1)(theta)
#   (5-8)  b_0       = -infinity   =>  P*_0     == 1
#   (5-9)  b_(m+1)   = +infinity   =>  P*_(m+1) == 0
#
# (5-4) telescopes over the categories, and with (5-8)/(5-9) as the boundary
# terms the sum collapses to P*_0 - P*_(m+1) = 1 - 0 = 1. That holds for the
# normal ogive and the logistic alike, because it depends only on the
# boundary conditions and the differencing, not on the link function -- which
# matters, since morie's _grm_category_probs uses the logistic form.
#
# This is the property that would break first if the boundary rows of `cum`
# were ever mis-set, or if the clip that guards log(0) were widened enough to
# distort the mass. Neither shows up in a "result has item_params" assertion.
# ---------------------------------------------------------------------------


def test_grm_category_probs_sum_to_one():
    """Category probabilities partition the probability mass at every theta.

    Samejima (1969) Ch 5, eqs (5-4) with (5-8) and (5-9), p.23-24.
    """
    import numpy as np

    from morie.fn.irtgr import _grm_category_probs

    theta = np.linspace(-4.0, 4.0, 81)
    for a, thresholds in [
        (1.0, [-1.0, 0.0, 1.0]),
        (0.5, [-2.5, 1.75]),
        (2.3, [-0.4, -0.1, 0.6, 2.2]),
    ]:
        probs = _grm_category_probs(theta, a, thresholds)
        assert probs.shape == (theta.size, len(thresholds) + 1)
        total = probs.sum(axis=1)
        # 1e-9 not exact equality: the implementation clips each category to
        # [1e-10, 1.0] to keep log() finite in the EM step, which can perturb
        # the sum by at most n_categories * 1e-10.
        assert np.allclose(total, 1.0, atol=1e-9), (
            f"mass not conserved for a={a}, b={thresholds}: "
            f"max deviation {np.max(np.abs(total - 1.0)):.3e}"
        )
        assert np.all(probs > 0.0)


def test_grm_cumulative_boundaries_are_monotone():
    """P*_x is decreasing in x at fixed theta, per the ordering in (5-3).

    Samejima (1969) Ch 5, eq (5-3), p.23: b_(x+1) > b_x. With a > 0 the
    cumulative probabilities inherit that ordering, so the implied
    P*_x(theta) sequence must be non-increasing across categories. If a
    threshold vector were ever passed unsorted, this is what would catch it.
    """
    import numpy as np

    from morie.fn.irtgr import _grm_category_probs

    theta = np.linspace(-3.0, 3.0, 25)
    thresholds = [-1.2, 0.3, 1.4]
    probs = _grm_category_probs(theta, 1.4, thresholds)
    # P*_k = sum of category probabilities at or above k; must be non-increasing.
    cum = np.cumsum(probs[:, ::-1], axis=1)[:, ::-1]
    assert np.all(np.diff(cum, axis=1) <= 1e-12)
