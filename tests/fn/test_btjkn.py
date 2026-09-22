"""Tests for btjkn.boot_jackknife."""

from morie.fn import _array_core as np

from morie.fn.btjkn import boot_jackknife


def test_btjkn_basic():
    """Test basic functionality against the documented jackknife formula."""
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 100)

    def stat(s):
        return np.mean(s)

    result = boot_jackknife(x, stat)

    # The documented return is a mapping with the listed keys; RichResult
    # supports dict-like access.
    assert isinstance(result, dict)
    for key in ("leave_one_out", "estimate", "bias", "corrected",
                "variance", "se", "pseudovalues", "n",
                "smoothness_caveat", "method"):
        assert key in result, f"missing key {key!r}"

    # Independent recomputation of the jackknife outputs from the closed-form
    # formula in the docstring.  For stat == mean the jackknife variance is
    # EXACTLY s^2 / n, not merely asymptotically.
    n = x.shape[0]
    th_indep = float(np.mean(x))
    loo_indep = np.array([float(np.mean(x[i != np.arange(n)]))
                          for i in range(n)])
    m_indep = float(loo_indep.mean())
    bias_indep = (n - 1) * (m_indep - th_indep)
    var_indep = (n - 1) / n * float(np.sum((loo_indep - m_indep) ** 2))
    pseudo_indep = n * th_indep - (n - 1) * loo_indep

    assert result["n"] == n
    assert result["estimate"] == th_indep
    assert np.allclose(result["leave_one_out"], loo_indep)
    assert result["bias"] == bias_indep
    assert result["corrected"] == th_indep - bias_indep
    assert result["variance"] == var_indep
    assert result["se"] == float(np.sqrt(var_indep))
    assert np.allclose(result["pseudovalues"], pseudo_indep)

    # Closed-form sanity check: jackknife variance equals s^2 / n exactly
    # for the sample mean.
    s2 = float(np.var(x, ddof=1))
    assert np.isclose(result["variance"], s2 / n)


def test_btjkn_edge():
    """Test the documented n < 3 guard."""
    import pytest

    x = np.array([1.0, 2.0])

    def stat(s):
        return float(np.mean(s))

    with pytest.raises(ValueError):
        boot_jackknife(x, stat)
