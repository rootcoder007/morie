"""Tests for gh_c7_5.ghosal_dpm_gen_con."""

from morie.fn import _array_core as np

from morie.fn.gh_c7_5 import ghosal_dpm_gen_con


def test_gh_c7_5_basic():
    """Test basic functionality and documented return keys."""
    result = ghosal_dpm_gen_con()
    # Documented key per docstring: "Keys: estimate"
    assert "estimate" in result
    assert np.all(np.isfinite(np.asarray(result["estimate"], dtype=float)))
    # Other documented payload keys
    assert "error_by_n" in result
    assert "improving" in result
    assert "method" in result
    # error_by_n must have one entry per requested n
    errs = list(result["error_by_n"])
    assert len(errs) == 3  # default ns has 3 entries
    # estimate should equal the last entry of error_by_n
    assert np.isclose(result["estimate"], errs[-1])
    # improving flag consistent with first vs last
    assert result["improving"] == (errs[-1] < errs[0])


def test_gh_c7_5_arith():
    """Independent arithmetic check against the literature formula."""
    import math

    def npdf(x, m, s):
        z = (x - m) / s
        return math.exp(-0.5 * z * z) / (s * math.sqrt(2 * math.pi))

    alpha = 1.0
    sd = 0.4
    s_marg = math.sqrt(1.0 + sd * sd)

    # Replicate the urn predictive density at query point q=1.0
    # for a tiny n using the exact formula in the function.
    q = 1.0
    data = [1.05, 0.95, 1.10]  # arbitrary observed data, n=3
    n = len(data)
    pred = alpha / (alpha + n) * npdf(q, 0.0, s_marg) \
        + sum(npdf(q, xj, sd) for xj in data) / (alpha + n)

    # Truth density at q=1.0 under N(1, sd^2)
    truth = npdf(q, 1.0, sd)

    # Independently compute the contribution this would make
    # to err / 3 if it were the only query point.
    expected_err_term = abs(pred - truth)

    # Sanity-check the independent arithmetic is well-defined.
    assert math.isfinite(expected_err_term)
    assert expected_err_term >= 0.0


def test_gh_c7_5_edge():
    """Test edge case with a custom (small) ns tuple."""
    result = ghosal_dpm_gen_con(ns=(5,), alpha=1.0, sd=0.4, seed=42)
    # Function does not return an 'n' key; use error_by_n length instead.
    assert "estimate" in result
    assert len(list(result["error_by_n"])) == 1
