"""Tests for irtgr — graded response model."""

from morie.fn.irtgr import irtgr


def test_irtgr_basic():
    """GRM recovers a real parameter structure from ordered-category data.

    The generated version fed the 200x20 five-category `mapq_df` fixture in,
    which is 20 items x 4 thresholds through a pure-Python EM -- minutes, not
    seconds -- and it asserted only that the result had an attribute. The
    model contract is identical at n=40, k=3, and can be checked for real.
    """
    rows = []
    for i in range(40):
        lvl = i % 3
        rows.append([lvl, min(2, lvl + i % 2), max(0, lvl - i % 2)])
    rows[0] = [0, 0, 0]
    rows[1] = [2, 2, 2]

    result = irtgr(rows, n_quad=11, max_iter=15)

    assert result.model == "GRM"
    assert sorted(result.item_params) == ["item_0", "item_1", "item_2"]
    for name, par in result.item_params.items():
        assert 0.01 <= par["a"] <= 5.0, (name, par["a"])
        # Samejima (5-3): thresholds are strictly ordered, one fewer than the
        # three categories present in the data.
        assert len(par["thresholds"]) == 2, (name, par["thresholds"])
        assert par["thresholds"][0] < par["thresholds"][1], (name, par["thresholds"])
        assert all(-6.0 <= b <= 6.0 for b in par["thresholds"]), (name, par["thresholds"])

    assert len(result.theta) == 40
    assert result.fit["n"] == 40
    assert result.fit["k"] == 3
    assert result.fit["loglik"] < 0.0
    assert 1 <= result.fit["n_iter"] <= 15

    # A respondent in the top category on every item must come out above one
    # in the bottom category on every item: every operating characteristic is
    # increasing in theta (Samejima 1969, Ch 5).
    assert float(result.theta[1]) > float(result.theta[0])
    # theta is a function of the response pattern alone, so identical rows
    # must give identical estimates.
    for i in range(2, 40):
        for j in range(i + 1, 40):
            if rows[i] == rows[j]:
                assert abs(float(result.theta[i]) - float(result.theta[j])) < 1e-12


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
    from morie.fn import _array_core as np

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
    from morie.fn import _array_core as np

    from morie.fn.irtgr import _grm_category_probs

    theta = np.linspace(-3.0, 3.0, 25)
    thresholds = [-1.2, 0.3, 1.4]
    probs = _grm_category_probs(theta, 1.4, thresholds)
    # P*_k = sum of category probabilities at or above k; must be non-increasing.
    cum = np.cumsum(probs[:, ::-1], axis=1)[:, ::-1]
    assert np.all(np.diff(cum, axis=1) <= 1e-12)
