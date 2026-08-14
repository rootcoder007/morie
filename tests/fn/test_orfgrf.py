"""orfgrf -- Orthogonal Random Forest. Source: Oprescu, Syrgkanis &
Wu (2019) PMLR 97, 4932-4941, arXiv:1806.03467."""
import pytest

from morie.fn.orfgrf import (local_nuisance, orthogonal_moment,
                             orthogonal_random_forest)


def plm(n=200, seed=0, theta=2.0):
    from morie.fn import _array_core as np
    rng = np.random.default_rng(seed)
    Y, T, X, W = [], [], [], []
    for _ in range(n):
        x = float(rng.uniform())
        w1 = float(rng.normal())
        t = 0.7 * w1 + float(rng.normal(0.0, 1.0))
        Y.append(theta * t + 2.0 * w1 + float(rng.normal(0.0, 0.3)))
        T.append(t)
        X.append([x])
        W.append([w1])
    return Y, T, X, W


def test_moment_solves_to_the_closed_form():
    th, den = orthogonal_moment([1.0, 2.0, 3.0, 4.0],
                                [1.0, 1.0, 2.0, 2.0], [1.0] * 4)
    assert th == pytest.approx((1 + 2 + 6 + 8) / 10.0, abs=1e-15)
    assert den == pytest.approx(10.0, abs=1e-15)


def test_moment_is_invariant_to_weight_scale():
    a, _ = orthogonal_moment([1.0, 2.0], [1.0, 2.0], [1.0, 1.0])
    b, _ = orthogonal_moment([1.0, 2.0], [1.0, 2.0], [7.0, 7.0])
    assert a == pytest.approx(b, abs=1e-14)


def test_moment_reproduces_a_planted_slope_with_no_noise():
    tr = [1.0, 2.0, 3.0, -1.0]
    yr = [3.0 * v for v in tr]
    th, _ = orthogonal_moment(yr, tr, [1.0] * 4)
    assert th == pytest.approx(3.0, abs=1e-13)


def test_local_nuisance_fits_an_exact_linear_surface():
    W = [[0.0], [1.0], [2.0], [3.0]]
    y = [1.0, 3.0, 5.0, 7.0]                 # y = 1 + 2 w
    fitted, b = local_nuisance(y, W, [1.0] * 4)
    assert fitted[2] == pytest.approx(5.0, abs=1e-6)
    assert b[1] == pytest.approx(2.0, abs=1e-6)


def test_local_nuisance_excludes_the_named_row():
    W = [[0.0], [1.0], [2.0], [3.0]]
    y = [1.0, 3.0, 5.0, 99.0]
    _, kept = local_nuisance(y, W, [1.0] * 4, exclude=3)
    assert kept[1] == pytest.approx(2.0, abs=1e-6)


def test_orf_recovers_a_constant_treatment_effect():
    Y, T, X, W = plm(240, seed=1, theta=2.0)
    r = orthogonal_random_forest(Y, T, X, W, n_trees=30, min_leaf=20,
                                 seed=1, residualize="global")
    assert r["estimate"] == pytest.approx(2.0, abs=0.2)


def test_both_residualization_routes_run_and_are_labelled():
    Y, T, X, W = plm(120, seed=2)
    for route in ("local", "global"):
        r = orthogonal_random_forest(Y, T, X, W, n_trees=8,
                                     min_leaf=15, seed=0,
                                     residualize=route,
                                     leave_one_out=False)
        assert r["residualize"] == route
        assert len(r["theta"]) == 120


def test_zero_residual_treatment_is_refused():
    with pytest.raises(ValueError):
        orthogonal_moment([1.0, 2.0], [0.0, 0.0], [1.0, 1.0])


def test_empty_neighbourhood_is_refused():
    with pytest.raises(ValueError):
        local_nuisance([1.0, 2.0], [[0.0], [1.0]], [0.0, 0.0])


def test_unknown_route_is_refused():
    Y, T, X, W = plm(60)
    with pytest.raises(ValueError):
        orthogonal_random_forest(Y, T, X, W, residualize="kernel")


def test_too_few_observations_are_refused():
    Y, T, X, W = plm(60)
    with pytest.raises(ValueError):
        orthogonal_random_forest(Y[:5], T[:5], X[:5], W[:5])


def test_mismatched_lengths_are_refused():
    Y, T, X, W = plm(60)
    with pytest.raises(ValueError):
        orthogonal_random_forest(Y, T[:-1], X, W)


def test_mismatched_residual_and_weight_lengths_are_refused():
    with pytest.raises(ValueError):
        orthogonal_moment([1.0, 2.0], [1.0, 2.0], [1.0])
