"""Tests for the Sec 5.6 nonlinear-prediction family.

splgk -- lognormal kriging,      Sec 5.6.1, eq (5.54)
sptgk -- trans-Gaussian kriging, Sec 5.6.2, eqs (5.58)-(5.60)

Both correct a bias that the naive back-transform carries, so the tests
check the CORRECTION, not merely that a number comes out.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn._schab_krig import ordinary_kriging, simple_kriging
from morie.fn._schab_vario import semivariogram
from morie.fn.splgk import schabenberger_lognormal_kriging as splgk
from morie.fn.sptgk import anamorphosis, normal_scores
from morie.fn.sptgk import schabenberger_trans_gaussian_kriging as sptgk


def _sites(k=5):
    g = np.arange(float(k))
    return np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)


def _y(coords):
    return np.sin(coords[:, 0] * 0.6) + np.cos(coords[:, 1] * 0.4)


def _gamma(h):
    return semivariogram(h, 0.1, 1.0, 3.0, "exponential")


def test_ordinary_kriging_weights_sum_to_one():
    """The unbiasedness constraint of Sec 5.2.2.2 -- lambda'1 = 1 is what the
    Lagrange multiplier is there to enforce."""
    coords = _sites()
    _, _, lam, _ = ordinary_kriging(coords, _y(coords), np.array([2.3, 1.7]),
                                    _gamma)
    assert lam.sum() == pytest.approx(1.0, abs=1e-12)


def test_ordinary_kriging_variance_has_both_published_forms():
    """eq (5.22) gives sigma^2_ok two ways: lambda'gamma(s0) + m, and
    2 lambda'gamma(s0) - lambda'Gamma lambda. They must agree."""
    coords = _sites()
    target = np.array([2.3, 1.7])
    _, var, lam, m = ordinary_kriging(coords, _y(coords), target, _gamma)
    gmat = _gamma(np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1))
    g0 = _gamma(np.linalg.norm(coords - target, axis=1))
    assert var == pytest.approx(float(lam @ g0 + m), abs=1e-12)
    assert var == pytest.approx(float(2 * lam @ g0 - lam @ gmat @ lam), abs=1e-10)


def test_ordinary_kriging_reproduces_a_constant_field():
    coords = _sites()
    pred, _, _, _ = ordinary_kriging(coords, np.full(coords.shape[0], 7.0),
                                     np.array([2.3, 1.7]), _gamma)
    assert pred == pytest.approx(7.0, abs=1e-9)


def test_lognormal_correction_uses_the_kriging_variance_not_the_process_variance():
    """eq (5.54) corrects by sigma^2_sk/2. The process variance sigma^2_Y is
    larger by c'Sigma^-1 c -- the variance of the predictor -- so using it
    over-corrects, and by the most where kriging is most confident. This is
    the defect the module's earlier formula carried."""
    coords = _sites()
    z = np.exp(_y(coords))
    target = np.atleast_2d([2.3, 1.7])
    res = splgk(coords, z, target[0])
    _, var_arr, _ = simple_kriging(coords, np.log(z), target)
    var_sk = float(np.asarray(var_arr).ravel()[0])
    assert res["log_variance"] == pytest.approx(var_sk, abs=1e-12)
    assert res["bias_factor"] == pytest.approx(np.exp(0.5 * var_sk), rel=1e-12)
    assert res["prediction"] == pytest.approx(
        res["naive_prediction"] * res["bias_factor"], rel=1e-12)


def test_lognormal_prediction_exceeds_the_naive_back_transform():
    """exp{p_sk} is biased LOW: the correction factor exp{sigma^2_sk/2} is
    above one whenever the kriging variance is positive."""
    coords = _sites()
    res = splgk(coords, np.exp(_y(coords)), np.array([2.3, 1.7]))
    assert res["log_variance"] > 0.0
    assert res["prediction"] > res["naive_prediction"]


def test_lognormal_matches_the_aitchison_brown_moment():
    """The result the correction is built on: for Y ~ G(mu, s2),
    E[exp Y] = exp{mu + s2/2}. Checked against the native generator."""
    from morie.fn._rng import random_normal
    mu, s2 = 0.7, 0.6
    y = mu + np.sqrt(s2) * random_normal(400000, seed=17)
    assert np.exp(y).mean() == pytest.approx(np.exp(mu + 0.5 * s2), rel=0.01)


def test_lognormal_rejects_non_positive_data():
    coords = _sites(3)
    z = np.exp(_y(coords))
    z[0] = 0.0
    with pytest.raises(ValueError):
        splgk(coords, z, np.array([1.0, 1.0]))


def test_trans_gaussian_correction_is_equation_5_58():
    """p_tg = phi(p_ok) + phi''(mu_Y)/2 (sigma^2_ok - 2 m_Y). Assert the
    assembled quantity, including the SIGN of the Lagrange multiplier --
    a flipped m would shift every prediction with no other symptom."""
    coords = _sites()
    y = _y(coords)
    target = np.array([2.3, 1.7])
    res = sptgk(coords, y, target, np.exp, np.exp, np.exp, _gamma)
    pred_ok, var_ok, _, m = ordinary_kriging(coords, y, target, _gamma)
    expected = np.exp(pred_ok) + 0.5 * np.exp(y.mean()) * (var_ok - 2.0 * m)
    assert res["prediction"] == pytest.approx(float(expected), rel=1e-12)
    assert res["lagrange"] == pytest.approx(m, abs=1e-14)


def test_trans_gaussian_mspe_is_equation_5_59():
    """E[(p_tg - Z)^2] ~ [phi'(mu_Y)]^2 sigma^2_ok."""
    coords = _sites()
    y = _y(coords)
    target = np.array([2.3, 1.7])
    res = sptgk(coords, y, target, np.exp, np.exp, np.exp, _gamma)
    _, var_ok, _, _ = ordinary_kriging(coords, y, target, _gamma)
    assert res["mspe"] == pytest.approx(np.exp(y.mean()) ** 2 * var_ok, rel=1e-12)


def test_identity_transformation_needs_no_correction():
    """phi(y) = y has phi'' = 0, so (5.58) collapses to plain ordinary
    kriging. A correction appearing here would mean the second-derivative
    term is wired to something other than phi''."""
    coords = _sites()
    y = _y(coords)
    target = np.array([2.3, 1.7])
    res = sptgk(coords, y, target,
                lambda v: v, lambda v: 1.0, lambda v: 0.0, _gamma)
    pred_ok, _, _, _ = ordinary_kriging(coords, y, target, _gamma)
    assert res["correction"] == pytest.approx(0.0, abs=1e-15)
    assert res["prediction"] == pytest.approx(pred_ok, rel=1e-12)


def test_normal_scores_and_anamorphosis_invert_each_other():
    """eq (5.60): phi^-1(z) = Phi^-1(F(z)) and phi(y) = F^-1(Phi(y))."""
    z = np.exp(_y(_sites()))
    scores = normal_scores(z)
    assert np.allclose(np.sort(anamorphosis(z, scores)), np.sort(z))
    assert scores.mean() == pytest.approx(0.0, abs=0.05)
    assert scores.std(ddof=1) == pytest.approx(1.0, abs=0.15)


def test_normal_scores_are_monotone_and_finite():
    """A rank transform must preserve order, and no observation may map to
    an infinite score -- which the plain i/n plotting position would do to
    the largest value."""
    z = np.exp(_y(_sites()))
    scores = normal_scores(z)
    assert np.all(np.isfinite(scores))
    assert np.all(np.diff(scores[np.argsort(z)]) > 0)
