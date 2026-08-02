"""Mixed-model shelf: BLUE, BLUP and Henderson's equations.

Sources are Montesinos Lopez, Montesinos Lopez and Crossa (2022),
*Multivariate Statistical Machine Learning Methods for Genomic
Prediction*; section and equation numbers are the book's.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.bluerg import blue_gls


def mixed_design(seed=0, n=60, q=8, var_u=2.0, var_e=0.5,
                 beta=(1.0, 3.0)):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    Z = np.zeros((n, q))
    Z[np.arange(n), np.arange(n) % q] = 1.0
    u = rng.normal(size=q) * np.sqrt(var_u)
    y = X @ np.asarray(beta) + Z @ u + rng.normal(size=n) * np.sqrt(var_e)
    return y, X, Z, np.eye(q) * var_u, np.eye(n) * var_e, u


def test_gls_reduces_to_ols_when_v_is_the_identity():
    X = np.column_stack([np.ones(4), [0.0, 1.0, 2.0, 3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    out = blue_gls(y, X)
    assert out["beta"] == pytest.approx([1.0, 2.0])


def test_gls_weights_by_the_inverse_variance():
    # one observation with a huge variance must be discounted
    X = np.column_stack([np.ones(3), [0.0, 1.0, 2.0]])
    y = np.array([0.0, 1.0, 100.0])
    V = np.diag([1.0, 1.0, 1e6])
    slope_w = blue_gls(y, X, V=V)["beta"][1]
    slope_u = blue_gls(y, X)["beta"][1]
    assert abs(slope_w - 1.0) < abs(slope_u - 1.0)


def test_hendersons_equations_give_exactly_the_gls_estimator():
    # equation (2.2) and the GLS form are the same quantity; this
    # identity is the check that V = Z Sigma Z' + R was assembled right
    y, X, Z, Sg, R, _ = mixed_design(seed=1)
    out = blue_gls(y, X, Z=Z, Sigma=Sg, R=R)
    assert out["mme_matches_gls"] < 1e-9


def test_blue_recovers_the_true_fixed_effects():
    ests = []
    for s in range(40):
        y, X, Z, Sg, R, _ = mixed_design(seed=100 + s)
        ests.append(blue_gls(y, X, Z=Z, Sigma=Sg, R=R)["beta"])
    m = np.mean(ests, axis=0)
    assert abs(m[0] - 1.0) < 0.35
    assert abs(m[1] - 3.0) < 0.10


def test_blup_tracks_the_realised_random_effects():
    y, X, Z, Sg, R, u = mixed_design(seed=2, n=120, q=10)
    out = blue_gls(y, X, Z=Z, Sigma=Sg, R=R)
    assert float(np.corrcoef(out["blup"], u)[0, 1]) > 0.9


def test_blup_shrinks_toward_zero():
    # shrinkage is the whole point of calling the effect random
    y, X, Z, Sg, R, u = mixed_design(seed=3, n=80, q=8, var_u=0.05,
                                     var_e=4.0)
    out = blue_gls(y, X, Z=Z, Sigma=Sg, R=R)
    # with a tiny prior variance the BLUPs are pulled hard to zero
    assert float(np.max(np.abs(out["blup"]))) < float(np.max(np.abs(u))) + 1e-9
    assert out["blup_shrinkage"] < 1.0


def test_larger_prior_variance_shrinks_less():
    tight = mixed_design(seed=4, var_u=0.05, var_e=2.0)
    loose = mixed_design(seed=4, var_u=20.0, var_e=2.0)
    a = blue_gls(tight[0], tight[1], Z=tight[2], Sigma=tight[3], R=tight[4])
    b = blue_gls(loose[0], loose[1], Z=loose[2], Sigma=loose[3], R=loose[4])
    assert a["blup_shrinkage"] < b["blup_shrinkage"]


def test_rank_deficiency_is_flagged_not_hidden():
    # cell-means design with an intercept: X is rank 2, not 3
    X = np.column_stack([np.ones(6), [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
    out = blue_gls(np.arange(6.0), X)
    assert out["rank"] == 2
    assert out["rank_deficient"] is True
    assert "not identified" in out["rank_note"]


def test_estimability_separates_contrasts_from_individual_effects():
    # b1 - b2 is estimable; b1 on its own is not
    X = np.column_stack([np.ones(6), [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]])
    K = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, 0.0]])
    out = blue_gls(np.arange(6.0), X, K=K)
    assert list(out["estimable"]) == [True, False]


def test_full_rank_makes_every_coefficient_estimable():
    X = np.column_stack([np.ones(5), np.arange(5.0)])
    K = np.eye(2)
    out = blue_gls(np.arange(5.0), X, K=K)
    assert out["rank_deficient"] is False
    assert all(out["estimable"])


def test_standard_errors_are_the_gls_ones():
    y, X, Z, Sg, R, _ = mixed_design(seed=5)
    out = blue_gls(y, X, Z=Z, Sigma=Sg, R=R)
    V = Z @ Sg @ Z.T + R
    Vi = np.linalg.pinv(V)
    cov = np.linalg.pinv(X.T @ Vi @ X)
    assert out["se"] == pytest.approx(np.sqrt(np.diag(cov)))


def test_ignoring_the_random_effect_understates_the_standard_error():
    # treating clustered data as independent is the classic error
    y, X, Z, Sg, R, _ = mixed_design(seed=6, n=100, q=5, var_u=4.0,
                                     var_e=0.5)
    correct = blue_gls(y, X, Z=Z, Sigma=Sg, R=R)["se"][0]
    naive = blue_gls(y, X)["se"][0]
    assert naive < correct


def test_input_validation():
    X = np.column_stack([np.ones(5), np.arange(5.0)])
    y = np.arange(5.0)
    with pytest.raises(ValueError, match="V must be"):
        blue_gls(y, X, V=np.eye(3))
    with pytest.raises(ValueError, match="Sigma must be"):
        blue_gls(y, X, Z=np.eye(5), Sigma=np.eye(2))
    with pytest.raises(ValueError, match="K must have"):
        blue_gls(y, X, K=np.ones((5, 5)))
