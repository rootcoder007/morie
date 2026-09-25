"""Tests for sptgk.schabenberger_trans_gaussian_kriging."""

from morie.fn import _array_core as np

from morie.fn.sptgk import (anamorphosis, normal_scores,
                            schabenberger_trans_gaussian_kriging)

COORDS = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
Z = [1.0, 2.0, 3.0, 4.0]
TARGET = [0.5, 0.5]


def _gamma(h):
    """Exponential semivariogram, unit sill and unit practical range."""
    return 1.0 - np.exp(-np.asarray(h, dtype=float))


def test_sptgk_basic():
    """The identity transform must reduce (5.58) to plain ordinary kriging."""
    result = schabenberger_trans_gaussian_kriging(
        COORDS, Z, TARGET,
        phi=lambda y: y, dphi=lambda y: 1.0, d2phi=lambda y: 0.0,
        semivariogram_fn=_gamma)

    # The second derivative is zero, which kills the (5.58) bias correction.
    assert abs(result["correction"]) < 1e-12
    assert abs(result["prediction"] - result["naive_prediction"]) < 1e-12
    # The first derivative is one, so (5.59) is the plain kriging variance.
    assert abs(result["mspe"] - result["kriging_variance"]) < 1e-12
    assert result["kriging_variance"] > 0.0
    assert abs(result["mu_y"] - 2.5) < 1e-12
    # The target is the centre of a symmetric square, so the ordinary
    # kriging weights are all 1/4 and the prediction is the sample mean.
    assert abs(result["prediction"] - 2.5) < 1e-9


def test_sptgk_quadratic_transform_correction():
    """With phi(y) = y**2 the correction is exactly (sigma2_ok - 2 m)."""
    result = schabenberger_trans_gaussian_kriging(
        COORDS, Z, TARGET,
        phi=lambda y: y * y, dphi=lambda y: 2.0 * y, d2phi=lambda y: 2.0,
        semivariogram_fn=_gamma)

    var_ok = result["kriging_variance"]
    m = result["lagrange"]
    assert abs(result["correction"] - (var_ok - 2.0 * m)) < 1e-12
    assert abs(result["naive_prediction"] - 2.5 ** 2) < 1e-9
    assert abs(result["prediction"]
               - (result["naive_prediction"] + result["correction"])) < 1e-12
    # (5.59) with dphi(mu_y) = 2 * 2.5 = 5.
    assert abs(result["mspe"] - 25.0 * var_ok) < 1e-9


def test_sptgk_edge():
    """normal_scores and anamorphosis are inverse on the observed support."""
    z = [3.0, 1.0, 4.0, 1.5, 5.0]
    scores = [float(v) for v in np.asarray(normal_scores(z)).ravel()]
    assert len(scores) == 5
    # Scores follow the data order and are the (i - 1/2)/n normal quantiles.
    assert scores[1] < scores[3] < scores[0] < scores[2] < scores[4]
    # n = 5 is symmetric: the middle score is 0 and the ends are opposite.
    assert abs(scores[0]) < 1e-12
    assert abs(scores[1] + scores[4]) < 1e-12

    back = [float(v) for v in np.asarray(anamorphosis(z, scores)).ravel()]
    for got, want in zip(back, z):
        assert abs(got - want) < 1e-9

    # A single observation has score 0 and maps back to itself.
    assert abs(float(np.asarray(normal_scores([7.0])).ravel()[0])) < 1e-12
    assert abs(float(np.asarray(anamorphosis([7.0], [0.0])).ravel()[0]) - 7.0) < 1e-12

    try:
        normal_scores([])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for empty z")
