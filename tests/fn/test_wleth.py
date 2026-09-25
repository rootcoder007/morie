"""Tests for wleth.weighted_likelihood_theta (Warm 1989)."""

import math

import pytest

from morie.fn.wleth import weighted_likelihood_theta

# eight 1PL items with difficulties symmetric about zero
B = [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]


def _p(theta, a, b, c):
    return c + (1.0 - c) / (1.0 + math.exp(-a * (theta - b)))


def _objective(theta, y, a, b, c):
    """log L(theta) + log sqrt(I(theta)), the quantity being maximised."""
    ll = 0.0
    info = 0.0
    for yi, ai, bi, ci in zip(y, a, b, c):
        p = min(max(_p(theta, ai, bi, ci), 1e-12), 1 - 1e-12)
        ll += yi * math.log(p) + (1.0 - yi) * math.log(1.0 - p)
        star = (p - ci) / max(1.0 - ci, 1e-12)
        dp = ai * (1.0 - ci) * star * (1.0 - star)
        info += dp * dp / (p * (1.0 - p))
    return ll + 0.5 * math.log(info), ll, info


def _ones(n):
    return [1.0] * n


def _zeros(n):
    return [0.0] * n


def test_wleth_basic():
    """The reported theta maximises the weighted objective, and the
    payload's information, se and weight term agree with it."""
    y = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    res = weighted_likelihood_theta(y, b=B)

    assert res["n_items"] == 8
    assert res["bias_corrected"] is True
    assert res["finite_for_perfect_patterns"] is True

    th = res["theta"]
    assert math.isfinite(th)
    # b is symmetric about 0 and so is this response pattern, so the
    # estimate must sit at the centre of the scale.
    assert th == pytest.approx(0.0, abs=1e-6)

    a, c = _ones(8), _zeros(8)
    obj, ll, info = _objective(th, y, a, B, c)
    assert res["information"] == pytest.approx(info, rel=1e-6)
    assert res["se"] == pytest.approx(1.0 / math.sqrt(info), rel=1e-6)
    assert res["weight_term"] == pytest.approx(0.5 * math.log(info), rel=1e-6)
    assert res["loglik"] == pytest.approx(ll, abs=1e-5)

    # it really is a maximum of the weighted objective
    for h in (0.02, 0.2, 1.0):
        assert _objective(th + h, y, a, B, c)[0] < obj + 1e-9
        assert _objective(th - h, y, a, B, c)[0] < obj + 1e-9


def test_wleth_is_monotone_in_the_number_correct():
    """More correct answers means a higher ability estimate."""
    counts = [0, 2, 4, 6, 8]
    thetas = []
    for k in counts:
        y = [1.0] * k + [0.0] * (8 - k)
        thetas.append(weighted_likelihood_theta(y, b=B)["theta"])
    assert all(thetas[i] < thetas[i + 1] for i in range(len(counts) - 1))
    # the all-wrong and all-correct estimates bracket the rest symmetrically
    assert thetas[0] == pytest.approx(-thetas[-1], abs=1e-6)


def test_wleth_perfect_patterns_stay_finite_where_ml_does_not():
    """sqrt(I) vanishes in both tails, so the weighted objective turns
    over: the ML estimate diverges but the WLE does not."""
    from morie.fn.mleth import mle_theta_estimator

    for y in ([1.0] * 8, [0.0] * 8):
        res = weighted_likelihood_theta(y, b=B)
        th = res["theta"]
        assert math.isfinite(th)
        assert -6.0 < th < 6.0            # strictly interior, not a bound hit
        assert res["se"] > 0.0 and math.isfinite(res["se"])
        # ML has no finite maximiser here, so there is nothing to compare to
        assert mle_theta_estimator(y, b=B)["finite"] is False
        assert res["vs_ml"] is None

        a, c = _ones(8), _zeros(8)
        obj = _objective(th, y, a, B, c)[0]
        for h in (0.05, 0.5, 2.0):
            assert _objective(th + h, y, a, B, c)[0] < obj + 1e-9
            assert _objective(th - h, y, a, B, c)[0] < obj + 1e-9

    # and the two perfect patterns are mirror images
    hi = weighted_likelihood_theta([1.0] * 8, b=B)["theta"]
    lo = weighted_likelihood_theta([0.0] * 8, b=B)["theta"]
    assert hi == pytest.approx(-lo, abs=1e-6)
    assert hi > 0.0 > lo


def test_wleth_reports_its_offset_from_ml_for_ordinary_patterns():
    """vs_ml is theta_WLE - theta_ML, and the correction shrinks the
    estimate toward the middle of the scale."""
    from morie.fn.mleth import mle_theta_estimator

    y = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]     # 7 of 8 correct
    res = weighted_likelihood_theta(y, b=B)
    ml = mle_theta_estimator(y, b=B)
    assert ml["finite"] is True
    assert res["vs_ml"] == pytest.approx(res["theta"] - ml["theta"], rel=1e-9,
                                         abs=1e-12)
    # the bias correction pulls a high ML estimate back down
    assert res["theta"] < ml["theta"]
    assert abs(res["theta"]) < abs(ml["theta"])


def test_wleth_edge():
    """Discrimination and guessing parameters, and the rejections."""
    y = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    a = [1.4] * 8
    c = [0.2] * 8
    res = weighted_likelihood_theta(y, a=a, b=B, c=c)
    obj, ll, info = _objective(res["theta"], y, a, B, c)
    assert res["information"] == pytest.approx(info, rel=1e-6)
    assert res["loglik"] == pytest.approx(ll, abs=1e-5)
    for h in (0.05, 0.5):
        assert _objective(res["theta"] + h, y, a, B, c)[0] < obj + 1e-9
        assert _objective(res["theta"] - h, y, a, B, c)[0] < obj + 1e-9

    # a sharper test carries more information, so a smaller standard error
    flat = weighted_likelihood_theta(y, a=[0.5] * 8, b=B)
    sharp = weighted_likelihood_theta(y, a=[2.0] * 8, b=B)
    assert sharp["information"] > flat["information"]
    assert sharp["se"] < flat["se"]

    # a narrower search interval is honoured
    boxed = weighted_likelihood_theta([1.0] * 8, b=B, bounds=(-1.0, 1.0))
    assert -1.0 <= boxed["theta"] <= 1.0

    # single item, still finite thanks to the weight
    one = weighted_likelihood_theta([1.0], b=[0.0])
    assert one["n_items"] == 1 and math.isfinite(one["theta"])

    with pytest.raises(ValueError):
        weighted_likelihood_theta([0.0, 0.5, 1.0], b=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        weighted_likelihood_theta(y)                      # b is required
    with pytest.raises(ValueError):
        weighted_likelihood_theta(y, b=B[:4])             # length mismatch
    with pytest.raises(ValueError):
        weighted_likelihood_theta(y, a=[1.0] * 4, b=B)
