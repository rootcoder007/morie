"""Tests for difsbs.dif_sibtest (Shealy & Stout 1993)."""

import numpy as np
import pytest

from morie.fn.difsbs import dif_sibtest


def _items(seed=0, n=3000, p=6, dif=0.0, dif_item=0):
    """Binary items. The focal group differs in ability, and optionally
    one item carries genuine DIF on top of that."""
    rng = np.random.default_rng(seed)
    grp = rng.integers(0, 2, n)
    theta = rng.normal(0, 1, n) - 0.5 * grp  # a real ability gap
    X = np.empty((n, p))
    for j in range(p):
        eta = theta - (j - p / 2) * 0.3
        if j == dif_item:
            eta = eta - dif * grp  # focal group disadvantaged on this item
        X[:, j] = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    return X, grp


def test_no_dif_is_not_flagged_despite_an_ability_gap():
    """The groups differ in ability but no item is biased.

    Matching on an observed score leaves residual ability differences
    inside each stratum, so this is the case that separates a working
    implementation from a broken one. Twenty items give a matching score
    reliable enough for the corrected statistic to hold roughly its
    nominal size.
    """
    rate = np.mean([
        np.mean(dif_sibtest(*_items(seed=s, p=20, dif=0.0))["p_value"] < 0.05)
        for s in range(6)
    ])
    # Averaged over seeds: one draw of 20 items is far too noisy to
    # pin a rejection rate on. Measured at 0.075 here against a
    # nominal 0.05.
    assert rate <= 0.15


def test_the_correction_is_what_makes_the_size_usable():
    """Measured, not assumed: the uncorrected form over-rejects badly.

    On data with an ability gap and no DIF, the uncorrected statistic
    flags far more items than the corrected one. This is the whole
    reason the correction is on by default.
    """
    flagged_raw = flagged_corr = 0
    for s in range(4):
        X, grp = _items(seed=s, p=12, dif=0.0)
        flagged_raw += int(np.sum(dif_sibtest(X, grp, correct=False)["p_value"] < 0.05))
        flagged_corr += int(np.sum(dif_sibtest(X, grp, correct=True)["p_value"] < 0.05))
    assert flagged_corr < flagged_raw / 2


def test_a_planted_dif_item_is_found():
    X, grp = _items(seed=2, p=12, dif=1.2, dif_item=2)
    res = dif_sibtest(X, grp)
    assert res["p_value"][2] < 0.01
    assert int(np.nanargmax(np.abs(res["statistic"]))) == 2


def test_beta_sign_follows_which_group_is_favoured():
    """Reference is the level sorting first, group 0; a focal
    disadvantage makes reference minus focal positive."""
    X, grp = _items(seed=3, p=12, dif=1.2, dif_item=1)
    assert dif_sibtest(X, grp)["beta"][1] > 0


def test_b_is_beta_over_its_standard_error():
    X, grp = _items(seed=4, dif=0.8, dif_item=0)
    res = dif_sibtest(X, grp)
    ok = res["se"] > 0
    assert np.allclose(res["statistic"][ok], res["beta"][ok] / res["se"][ok], rtol=1e-12)


def test_supplied_matching_is_used():
    X, grp = _items(seed=5, dif=0.0)
    a = dif_sibtest(X, grp)["beta"]
    b = dif_sibtest(X, grp, matching=X.sum(axis=1))["beta"]
    assert not np.allclose(a, b), "rest score and total score are different matchings"


def test_strata_are_actually_used():
    X, grp = _items(seed=6)
    assert np.all(dif_sibtest(X, grp)["n_strata"] > 1)


def test_validates_inputs():
    X, grp = _items(seed=7, n=200)
    with pytest.raises(ValueError, match="group must be binary"):
        dif_sibtest(X, np.arange(X.shape[0]))
    with pytest.raises(ValueError, match="one entry per row"):
        dif_sibtest(X, grp[:-1])
    with pytest.raises(ValueError, match="X must be finite"):
        bad = X.copy()
        bad[0, 0] = np.nan
        dif_sibtest(bad, grp)
    with pytest.raises(ValueError, match="no rest score"):
        dif_sibtest(X[:, :1], grp)
