"""Book-certified tests for the Schabenberger & Gotway kriging family.

The decisive property is exact interpolation: simple kriging predicting
at an observed location must return that observation with zero variance,
because sigma becomes a column of Sigma. The book calls this "honouring
the data" (p. 224). It is a structural identity, so it holds to machine
precision rather than approximately.

Schabenberger, O. & Gotway, C. A. (2005). Ch. 5.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spskrg import schabenberger_simple_kriging as simple_kriging
from morie.fn.spblup import schabenberger_blup as blup
from morie.fn.spkwt import schabenberger_kriging_weights as kriging_weights
from morie.fn.spkfnn import schabenberger_cross_validation_kriging as loo_cv
from morie.fn.spnsr import schabenberger_nugget_sill_range_effect as nsr_effect

CM = {"nugget": 0.0, "sill": 1.0, "range": 2.0, "model": "exponential"}


def _field(seed=0, n=25):
    rng = np.random.default_rng(seed)
    coords = rng.random((n, 2)) * 5.0
    return coords, np.sin(coords[:, 0]) + np.cos(coords[:, 1])


def test_simple_kriging_is_an_exact_interpolator():
    """Predicting at an observed location returns it exactly (p. 224)."""
    coords, z = _field()
    r = simple_kriging(coords, z, coords, CM)
    np.testing.assert_allclose(r["prediction"], z, atol=1e-12)
    assert np.max(r["variance"]) < 1e-12


def test_simple_kriging_weight_on_the_point_itself_is_one():
    coords, z = _field()
    w = simple_kriging(coords, z, coords, CM)["weights"]
    np.testing.assert_allclose(np.diag(w), 1.0, atol=1e-10)


def test_simple_kriging_matches_the_closed_form(): 
    """p = mu + sigma' Sigma^-1 (Z - mu), sigma^2 - sigma' Sigma^-1 sigma."""
    from morie.fn._schab_krig import cov_from_model, _dist

    coords, z = _field(seed=5, n=12)
    tgt = np.array([[2.0, 2.0]])
    mu = 0.3
    Sigma = cov_from_model(_dist(coords, coords), CM)
    sig = cov_from_model(_dist(coords, tgt), CM)[:, 0]
    s2 = float(cov_from_model(np.zeros(1), CM)[0])
    lam = np.linalg.solve(Sigma, sig)
    r = simple_kriging(coords, z, tgt, CM, mu=mu)
    assert r["prediction"][0] == pytest.approx(mu + lam @ (z - mu))
    assert r["variance"][0] == pytest.approx(s2 - sig @ lam)


def test_blup_weights_sum_to_one():
    """Unbiasedness with an unknown mean forces sum(lambda) = 1."""
    coords, z = _field()
    tgt = np.array([[1.0, 1.0], [3.0, 2.0]])
    w = blup(coords, z, tgt, CM)["weights"]
    np.testing.assert_allclose(w.sum(axis=0), 1.0, atol=1e-10)


def test_blup_also_honours_the_data():
    coords, z = _field()
    r = blup(coords, z, coords, CM)
    np.testing.assert_allclose(r["prediction"], z, atol=1e-10)


def test_blup_variance_is_never_below_simple_kriging():
    """Not knowing the mean cannot help: the BLUP pays for the constraint."""
    coords, z = _field()
    tgt = np.array([[1.3, 2.7], [4.0, 0.5]])
    sk = simple_kriging(coords, z, tgt, CM)["variance"]
    ok = blup(coords, z, tgt, CM)["variance"]
    assert np.all(ok >= sk - 1e-12)


def test_ordinary_kriging_weights_sum_to_one_exactly():
    Sigma = np.eye(3) * 2.0
    sig = np.array([1.0, 0.5, 0.25])
    r = kriging_weights(Sigma, sig, unbiased=True)
    assert r["weight_sum"] == pytest.approx(1.0, abs=1e-12)
    assert r["lagrange"] is not None


def test_simple_kriging_weights_solve_sigma_lambda_equals_sigma():
    rng = np.random.default_rng(2)
    A = rng.random((4, 4))
    Sigma = A @ A.T + 4 * np.eye(4)
    sig = rng.random(4)
    lam = kriging_weights(Sigma, sig)["weights"]
    np.testing.assert_allclose(Sigma @ lam, sig, atol=1e-10)


def test_kriging_weight_input_validation():
    with pytest.raises(ValueError, match="square"):
        kriging_weights(np.ones((2, 3)), np.ones(2))
    with pytest.raises(ValueError, match="one entry per observation"):
        kriging_weights(np.eye(3), np.ones(2))


def test_loo_cv_residuals_are_not_identically_zero():
    """In-sample residuals ARE zero because kriging interpolates; the
    point of cross-validation is that leave-one-out residuals are not."""
    coords, z = _field()
    r = loo_cv(coords, z, CM)
    assert r["mspe"] > 0
    assert r["rmspe"] == pytest.approx(np.sqrt(r["mspe"]))
    assert abs(r["me"]) < 0.5 * r["rmspe"]      # roughly unbiased
    assert r["residuals"].size == z.size


def test_loo_cv_needs_enough_points():
    with pytest.raises(ValueError, match="at least 3 points"):
        loo_cv(np.zeros((2, 2)), np.zeros(2), CM)


def test_sill_scales_the_variance_but_not_the_prediction():
    """The sill is a pure variance factor: weights, and so the
    prediction, are invariant to it (Sec. 5.2.3)."""
    a = nsr_effect(0.0, 1.0, 1.0)
    b = nsr_effect(0.0, 7.0, 1.0)
    assert b["prediction"] == pytest.approx(a["prediction"])
    assert b["variance"] / a["variance"] == pytest.approx(7.0)
    np.testing.assert_allclose(a["weights"], b["weights"], atol=1e-12)


def test_a_pure_nugget_collapses_the_prediction_to_the_mean():
    """With no spatially correlated component the weights are equal and
    the predictor is the mean -- maximal smoothing."""
    r = nsr_effect(nugget=1.0, sill=0.0, range=1.0)
    assert r["prediction"] == pytest.approx(r["mean"])
    assert r["weight_spread"] == pytest.approx(0.0, abs=1e-12)


def test_a_larger_nugget_flattens_the_weights():
    spreads = [nsr_effect(nug, 1.0, 1.0)["weight_spread"]
               for nug in (0.0, 0.5, 2.0, 10.0)]
    assert spreads[0] > spreads[1] > spreads[2] > spreads[3]


def test_nsr_input_validation():
    with pytest.raises(ValueError, match="`range` must be"):
        nsr_effect(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="must be >= 0"):
        nsr_effect(-1.0, 1.0, 1.0)
