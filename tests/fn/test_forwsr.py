"""Tests for morie.fn.forwsr -- the forward search.

The anchors are the two properties the method exists for: a clean
data set produces no jump, and a contaminated one holds its outliers
outside the subset until the end and flags them. A test that only
asserts the return is a dict would pass on a stub.
"""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.forwsr import (consistency_factor, forward_plot,
                             forward_search,
                             forward_search_regression, lms_start,
                             ols_fit)


def _clean(n=60):
    """y = 2 + 3 x1 - x2 plus a tiny wobble; X carries its own
    intercept column, since the search takes the design as given."""
    rng = np.random.default_rng(7)
    X, y = [], []
    for i in range(n):
        x1 = (i % 7) / 7.0
        x2 = ((i * 3) % 11) / 11.0
        X.append([1.0, x1, x2])
        y.append(2.0 + 3.0 * x1 - x2 + float(rng.normal(0.0, 0.05)))
    return X, y


def test_ols_fit_recovers_the_generating_coefficients():
    X, y = _clean()
    f = ols_fit(X, y)
    assert f["beta"][0] == pytest.approx(2.0, abs=5e-2)
    assert f["beta"][1] == pytest.approx(3.0, abs=5e-2)
    assert f["beta"][2] == pytest.approx(-1.0, abs=5e-2)


def test_search_ends_at_the_full_sample_and_grows_by_one():
    X, y = _clean()
    steps = forward_search(X, y)
    assert steps[-1]["m"] == len(y)
    ms = [s["m"] for s in steps]
    assert ms == list(range(ms[0], len(y) + 1))
    # the last step is least squares on everything
    assert steps[-1]["beta"] == pytest.approx(ols_fit(X, y)["beta"])


def test_clean_data_raises_no_flag():
    X, y = _clean()
    r = forward_search_regression(X, y)
    assert r["n_flagged"] == 0


def test_two_outliers_do_not_mask_each_other():
    """The point of the method: both are found, and both enter last."""
    X, y = _clean()
    bad = (11, 37)
    for i in bad:
        y[i] += 25.0
    r = forward_search_regression(X, y)
    assert r["n_flagged"] >= len(bad)
    # neither contaminated row is in the subset until the closing steps
    for s in r["steps"]:
        if s["m"] <= len(y) - len(bad):
            assert all(i not in s["subset"] for i in bad)


def test_lms_start_avoids_the_contaminated_rows():
    X, y = _clean()
    y[11] += 25.0
    y[37] += 25.0
    sub = lms_start(X, y)["subset"]
    assert 11 not in sub and 37 not in sub


def test_sigma_steps_up_when_contamination_enters():
    X, y = _clean()
    y[11] += 25.0
    y[37] += 25.0
    sig = [s["sigma"] for s in forward_search(X, y)]
    # the largest single-step jump is at the end, not in the middle
    gaps = [sig[i + 1] - sig[i] for i in range(len(sig) - 1)]
    assert gaps.index(max(gaps)) >= len(gaps) - 3


def test_forward_plot_rejects_an_unmonitored_key():
    X, y = _clean()
    steps = forward_search(X, y)
    p = forward_plot(steps)
    assert len(p["m"]) == len(steps)
    with pytest.raises(ValueError, match="not monitored"):
        forward_plot(steps, key="nonsense")


def test_a_short_starting_subset_is_refused():
    X, y = _clean()
    with pytest.raises(ValueError, match="at least"):
        forward_search(X, y, start=[0, 1])


def test_deletion_residual_is_finite_once_the_scale_is_estimable():
    """At m = p the subset has zero residual df, so sigma is 0 and the
    ratio is undefined; from m = p + 1 on it must be a real number."""
    X, y = _clean()
    p = len(X[0])
    for s in forward_search(X, y):
        v = s["min_deletion_residual"]
        if p < s["m"] < len(y):
            assert v == v and math.isfinite(v)
    assert forward_search(X, y)[-1]["min_deletion_residual"] != \
        forward_search(X, y)[-1]["min_deletion_residual"]  # nan at m = n


def test_consistency_factor_flattens_the_scale():
    """The anchor for the correction: without it the raw scale climbs
    by an order of magnitude across a CLEAN search; with it the scale
    sits at the true sigma from early on."""
    rng = np.random.default_rng(11)
    n, sigma = 200, 0.05
    X, y = [], []
    for i in range(n):
        x1 = (i % 7) / 7.0
        x2 = ((i * 3) % 11) / 11.0
        X.append([1.0, x1, x2])
        y.append(2.0 + 3.0 * x1 - x2 + float(rng.normal(0.0, sigma)))
    steps = forward_search(X, y)
    raw = [s["sigma"] for s in steps if s["m"] >= 20]
    cor = [s["sigma_corrected"] for s in steps if s["m"] >= 20]
    assert max(raw) / min(raw) > 8.0            # the bias is real
    assert max(cor) / min(cor) < 1.5            # the fix removes it
    for v in cor:
        assert abs(v - sigma) < 0.25 * sigma


def test_consistency_factor_is_one_at_the_full_sample():
    assert consistency_factor(50, 50) == 1.0
    assert 0.0 < consistency_factor(25, 50) < 1.0
