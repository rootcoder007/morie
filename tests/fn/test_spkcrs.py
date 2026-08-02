"""spkcrs -- cross K-function, Schabenberger & Gotway Sec. 3.4.4."""

from morie.fn import _array_core as np
import pytest

from morie.fn._schab_pp import ripley_weight, ripley_weights
from morie.fn.spkcrs import schabenberger_cross_k_function as ck

REG = (0.0, 0.0, 1.0, 1.0)
R = np.linspace(0.02, 0.25, 8)


def _pair(n=400, seed=11):
    rs = np.random.RandomState(seed)
    return rs.uniform(0, 1, (n, 2)), rs.uniform(0, 1, (n, 2)), rs


def test_ripley_weight_is_the_circumference_proportion():
    """Spot-check the exact weight against dense numerical integration."""
    rs = np.random.RandomState(2)
    for _ in range(10):
        p = rs.uniform(0.05, 0.95, 2)
        t = rs.uniform(0.05, 0.6)
        th = np.linspace(0, 2 * np.pi, 400001)[:-1]
        num = np.mean((p[0] + t * np.cos(th) >= 0) & (p[0] + t * np.cos(th) <= 1)
                      & (p[1] + t * np.sin(th) >= 0) & (p[1] + t * np.sin(th) <= 1))
        assert ripley_weight(p, REG, t) == pytest.approx(num, abs=5e-5)


def test_vectorised_weight_identical_to_scalar():
    rs = np.random.RandomState(6)
    p = rs.uniform(0.02, 0.98, (150, 2))
    t = rs.uniform(0.01, 0.7, 150)
    scal = np.array([ripley_weight(p[i], REG, t[i]) for i in range(150)])
    assert np.array_equal(scal, ripley_weights(p, REG, t))


def test_interior_circle_has_weight_one_and_zero_radius_too():
    assert ripley_weight((0.5, 0.5), REG, 0.4) == 1.0
    assert ripley_weight((0.9, 0.9), REG, 0.0) == 1.0


def test_independent_patterns_track_pi_h_squared():
    """p. 104: under independence K_ij(h) = pi h^2 regardless of the patterns.

    Monte-Carlo, so asserted at the measured n=400 band (worst of 20
    replicates was 0.156) rather than a taste threshold.
    """
    a, b, _ = _pair()
    res = ck(a, b, region=REG, r=R)
    rel = np.abs(res["estimate"] - np.pi * R ** 2) / (np.pi * R ** 2)
    assert rel.max() < 0.20
    assert np.abs(res["L_minus_h"]).max() < 0.02


def test_estimator_asymmetry_and_the_pooled_fix():
    """Khat_12 != Khat_21; Lotwick-Silverman's K* lies between them."""
    a, b, _ = _pair()
    res = ck(a, b, region=REG, r=R)
    assert not np.allclose(res["K_12"], res["K_21"])
    lo = np.minimum(res["K_12"], res["K_21"]) - 1e-12
    hi = np.maximum(res["K_12"], res["K_21"]) + 1e-12
    assert np.all((res["estimate"] >= lo) & (res["estimate"] <= hi))


def test_attraction_pushes_l_star_positive():
    a, _, rs = _pair()
    b = np.clip(a + rs.normal(0, 0.01, a.shape), 0.001, 0.999)
    lm = ck(a, b, region=REG, r=R)["L_minus_h"]
    assert lm.mean() > 0
    assert np.all(lm[2:] > 0)


def test_inhibition_pushes_l_star_negative():
    """Type-2 events forced away from type-1 events -> repulsion."""
    rs = np.random.RandomState(21)
    a = rs.uniform(0, 1, (250, 2))
    b = []
    while len(b) < 250:
        c = rs.uniform(0, 1, 2)
        if np.min(np.linalg.norm(a - c, axis=1)) > 0.06:
            b.append(c)
    lm = ck(a, np.array(b), region=REG, r=R)["L_minus_h"]
    assert lm[0] < 0 and lm.mean() < 0


def test_edge_correction_raises_the_estimate():
    a, b, _ = _pair(200)
    cor = ck(a, b, region=REG, r=R, correction="ripley")["K_12"]
    unc = ck(a, b, region=REG, r=R, correction="none")["K_12"]
    assert np.all(cor >= unc - 1e-12)


def test_random_labelling_is_a_different_null():
    """eq (3.10): the hypothesis switch returns D(h) = K_11 - K_22."""
    a, b, _ = _pair(200)
    res = ck(a, b, region=REG, r=R, hypothesis="random_labelling")
    assert "D" in res and "K_11" in res and "K_22" in res
    same = ck(a, a, region=REG, r=R, hypothesis="random_labelling")
    assert np.abs(same["D"]).max() < 1e-12


def test_supplied_intensities_are_reported_not_substituted():
    a, b, _ = _pair(150)
    res = ck(a, b, lambda1=999.0, lambda2=1.0, region=REG, r=R)
    assert res["lambda_1_supplied"] == 999.0
    assert res["lambda_1"] == pytest.approx(150.0)


def test_default_r_grid_is_generated():
    a, b, _ = _pair(100)
    res = ck(a, b, region=REG)
    assert len(res["r"]) > 0 and np.all(np.asarray(res["r"]) > 0)


def test_rejects_bad_input():
    a, b, _ = _pair(50)
    with pytest.raises(ValueError):
        ck(a, b, region=REG, r=[-0.1])
    with pytest.raises(ValueError):
        ck(a, b, region=REG, r=R, hypothesis="both")
    with pytest.raises(ValueError):
        ck(a, b, region=REG, r=R, correction="toroidal")
    with pytest.raises(ValueError):
        ck(np.empty((0, 2)), b, region=REG, r=R)
