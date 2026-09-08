"""cttitc: corrected item-total correlation (Nunnally & Bernstein 1994)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.cttitc import ctt_item_total_corr as itc


def test_cttitc_excludes_the_item_from_the_total():
    """Computed by hand against corr(item, rowsum - item)."""
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 5.0],
            [3.0, 4.0, 1.0],
            [4.0, 3.0, 2.0],
        ]
    )
    item = X[:, 0]
    rest = X.sum(axis=1) - item
    assert itc(X, 0)["estimate"] == pytest.approx(float(np.corrcoef(item, rest)[0, 1]))


def test_cttitc_correction_matters_for_independent_items():
    """With independent items the true item-rest correlation is ~0.

    The UNcorrected value is inflated to roughly 1/sqrt(k) because the item is
    inside its own total. This is the defect the correction exists for, so the
    test pins the gap rather than just the corrected number.
    """
    rng = np.random.default_rng(77)
    k = 5
    X = rng.standard_normal((4000, k))
    r = itc(X, 0)
    assert abs(r["estimate"]) < 0.05
    assert r["uncorrected"] == pytest.approx(1.0 / np.sqrt(k), abs=0.05)
    assert r["uncorrected"] > r["estimate"] + 0.3


def test_cttitc_detects_a_genuinely_good_item():
    """An item loading on the same latent factor as the rest correlates high."""
    rng = np.random.default_rng(83)
    n = 3000
    latent = rng.standard_normal(n)
    X = np.column_stack([latent + rng.standard_normal(n) * 0.5 for _ in range(6)])
    assert itc(X, 0)["estimate"] > 0.6


def test_cttitc_detects_a_reverse_keyed_item():
    """A reverse-scored item shows a strongly NEGATIVE corrected correlation.

    Finding this is the main practical use of the statistic.
    """
    rng = np.random.default_rng(89)
    n = 2000
    latent = rng.standard_normal(n)
    good = [latent + rng.standard_normal(n) * 0.5 for _ in range(5)]
    X = np.column_stack([-latent + rng.standard_normal(n) * 0.5] + good)
    assert itc(X, 0)["estimate"] < -0.5


def test_cttitc_reports_the_scale_geometry():
    rng = np.random.default_rng(91)
    r = itc(rng.standard_normal((50, 4)), 2)
    assert r["n"] == 50
    assert r["n_items"] == 4
    assert r["item_index"] == 2


def test_cttitc_rejects_a_single_item():
    """With one item, 'total minus the item' is empty."""
    with pytest.raises(ValueError, match="at least 2 items"):
        itc(np.zeros((10, 1)), 0)


def test_cttitc_rejects_a_constant_item_and_bad_index():
    X = np.column_stack([np.ones(10), np.arange(10.0), np.arange(10.0) ** 2])
    with pytest.raises(ValueError, match="zero variance"):
        itc(X, 0)
    with pytest.raises(ValueError, match="out of range"):
        itc(X, 5)
