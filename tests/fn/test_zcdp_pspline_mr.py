"""zCDP accounting, P-splines, and inverse-variance weighted MR."""
import importlib
import math

import pytest

Z = importlib.import_module("morie.fn.zfmech")
S = importlib.import_module("morie.fn.smfd")
M = importlib.import_module("morie.fn.mtr2sx")
Rng = importlib.import_module("morie.fn.survrsf")._Rng

rng = Rng(3)
N = 50
X = [i / (N - 1.0) for i in range(N)]
Y = [math.sin(6.0 * v) + 0.2 * (rng.next() - 0.5) for v in X]

BX = [0.10, 0.20, 0.15, 0.30]
SX = [0.01, 0.02, 0.015, 0.02]
BY = [0.05, 0.11, 0.07, 0.16]
SY = [0.01, 0.015, 0.012, 0.02]


# -------------------------------------------------------------- zfmech
@pytest.mark.parametrize("alpha", [1.5, 2.0, 7.0])
def test_the_gaussian_divergence_is_linear_in_alpha(alpha):
    d = Z.renyi_divergence_gaussian(0.0, 1.0, 2.0, alpha)
    assert d == pytest.approx(alpha * Z.zcdp_of_gaussian(1.0, 2.0))


def test_alpha_must_exceed_one():
    with pytest.raises(ValueError):
        Z.renyi_divergence_gaussian(0.0, 1.0, 2.0, 1.0)


def test_the_gaussian_mechanism_rho():
    assert Z.zcdp_of_gaussian(1.0, 2.0) == pytest.approx(0.125)
    assert Z.sigma_for_rho(1.0, 0.125) == pytest.approx(2.0)


def test_composition_is_additive():
    assert Z.compose([0.1, 0.2, 0.05])["rho"] == pytest.approx(0.35)


def test_group_privacy_is_quadratic():
    assert Z.group_privacy(0.1, 3)["rho"] == pytest.approx(0.9)
    assert Z.group_privacy(0.1, 1)["rho"] == pytest.approx(0.1)


def test_postprocessing_leaves_rho_alone():
    assert Z.postprocessing(0.3)["rho"] == pytest.approx(0.3)


def test_pure_dp_conversion():
    assert Z.from_pure_dp(0.5)["rho"] == pytest.approx(0.125)


def test_approx_dp_conversion_is_the_stated_formula():
    e = Z.to_approx_dp(0.1, 1e-6)["epsilon"]
    assert e == pytest.approx(0.1 + 2 * math.sqrt(0.1 * math.log(1e6)))


def test_the_conversions_do_not_round_trip():
    assert Z.round_trip(1.0, 1e-6)["inflation"] > 1.0


@pytest.mark.parametrize("call", [
    lambda: Z.to_approx_dp(0.1, 0.0),
    lambda: Z.to_approx_dp(0.1, 1.0),
    lambda: Z.zcdp_of_gaussian(1.0, 0.0),
    lambda: Z.group_privacy(0.1, 0),
    lambda: Z.compose([-0.1]),
])
def test_invalid_privacy_parameters_are_refused(call):
    with pytest.raises(ValueError):
        call()


def test_the_released_noise_has_the_promised_scale():
    g = Z.gaussian_mechanism(10.0, 1.0, 0.5, seed=1, n=3000)
    m = sum(g["release"]) / len(g["release"])
    sd = math.sqrt(sum((v - m) ** 2 for v in g["release"])
                   / len(g["release"]))
    assert sd == pytest.approx(g["sigma"], rel=0.08)


# ---------------------------------------------------------------- smfd
def test_the_basis_is_a_partition_of_unity():
    B = S.bspline_basis(X, S.knot_sequence(0.0, 1.0, 8, 3), 3)
    assert S.partition_of_unity(B)["ok"]


def test_a_cubic_touches_at_most_four_basis_functions():
    B = S.bspline_basis(X, S.knot_sequence(0.0, 1.0, 8, 3), 3)
    assert max(sum(1 for v in row if v > 0) for row in B) <= 4


def test_knot_sequence_validation():
    with pytest.raises(ValueError):
        S.knot_sequence(1.0, 0.0, 5, 3)
    with pytest.raises(ValueError):
        S.knot_sequence(0.0, 1.0, 0, 3)


def test_the_second_difference_matrix():
    D = S.difference_matrix(6, 2)
    assert len(D) == 4
    assert D[0] == [1.0, -2.0, 1.0, 0.0, 0.0, 0.0]


def test_the_penalty_order_must_fit_the_basis():
    with pytest.raises(ValueError):
        S.difference_matrix(2, 5)


def test_a_second_order_penalty_annihilates_a_line():
    D = S.difference_matrix(6, 2)
    line = [1.0 + 2.0 * k for k in range(6)]
    for row in D:
        assert sum(row[c] * line[c] for c in range(6)) == \
            pytest.approx(0.0, abs=1e-12)


def test_the_infinite_lambda_limit_is_the_ols_line():
    f = S.fit(X, Y, 10, 3, lam=1e12, order=2)
    mx = sum(X) / N
    my = sum(Y) / N
    b1 = (sum((X[i] - mx) * (Y[i] - my) for i in range(N))
          / sum((v - mx) ** 2 for v in X))
    b0 = my - b1 * mx
    for i in range(N):
        assert f["fitted"][i] == pytest.approx(b0 + b1 * X[i],
                                               abs=1e-4)


def test_a_first_order_penalty_gives_the_mean():
    f = S.fit(X, Y, 10, 3, lam=1e12, order=1)
    assert max(f["fitted"]) - min(f["fitted"]) < 1e-6
    assert f["fitted"][0] == pytest.approx(sum(Y) / N, abs=1e-4)


def test_the_effective_dimension_falls_with_lambda():
    a = S.fit(X, Y, 10, 3, 1e-4, 2)["effective_dimension"]
    b = S.fit(X, Y, 10, 3, 1.0, 2)["effective_dimension"]
    c = S.fit(X, Y, 10, 3, 1e12, 2)["effective_dimension"]
    assert a > b > c
    assert c == pytest.approx(2.0, abs=1e-3)


def test_leave_one_out_matches_the_hat_diagonal_shortcut():
    f = S.fit(X, Y, 8, 3, 1.0, 2)
    quick = S.cross_validation(f)["cv"]
    tot = 0.0
    for i in range(N):
        fi = S.fit(X[:i] + X[i + 1:], Y[:i] + Y[i + 1:], 8, 3, 1.0, 2)
        tot += (Y[i] - S.predict(fi, [X[i]])[0]) ** 2
    assert quick == pytest.approx(math.sqrt(tot / N), rel=0.1)


def test_a_negative_lambda_is_refused():
    with pytest.raises(ValueError):
        S.fit(X, Y, 10, 3, lam=-1.0)


def test_an_unknown_criterion_is_refused():
    with pytest.raises(ValueError):
        S.choose_lambda(X, Y, criterion="bic")


def test_choose_lambda_returns_the_grid_minimum():
    ch = S.choose_lambda(X, Y, lambdas=[0.01, 1.0, 100.0])
    assert ch["score"] == min(t["score"] for t in ch["trace"])


# -------------------------------------------------------------- mtr2sx
def test_ivw_equals_the_weighted_regression_through_the_origin():
    r = M.ivw(BX, SX, BY, SY, model="fixed")
    assert r["estimate"] == pytest.approx(r["regression_estimate"],
                                          abs=1e-14)
    assert r["se"] == pytest.approx(r["regression_se_fixed"],
                                    abs=1e-14)


def test_one_variant_gives_its_own_ratio():
    r = M.ivw([0.2], [0.02], [0.11], [0.015], model="fixed")
    assert r["estimate"] == pytest.approx(0.55)
    assert r["se"] == pytest.approx(0.075)


def test_the_ratio_estimates_are_beta_y_over_beta_x():
    assert M.ratio_estimates([0.2, 0.5], [0.1, 0.2]) == \
        pytest.approx([0.5, 0.4])


def test_fixed_and_multiplicative_share_the_point_estimate():
    a = M.ivw(BX, SX, BY, SY, model="fixed")
    b = M.ivw(BX, SX, BY, SY, model="multiplicative")
    assert a["estimate"] == pytest.approx(b["estimate"])


def test_underdispersion_does_not_shrink_the_interval():
    b = M.ivw(BX, SX, BY, SY, model="multiplicative")
    assert b["Q"] < b["df"]
    assert b["phi_multiplicative"] == 1.0
    assert b["se"] == pytest.approx(b["se_fixed"])


def test_heterogeneity_widens_the_multiplicative_interval_exactly():
    by = [0.05, 0.30, 0.02, 0.40]
    f = M.ivw(BX, SX, by, SY, model="fixed")
    m = M.ivw(BX, SX, by, SY, model="multiplicative")
    assert m["se"] == pytest.approx(f["se"]
                                    * math.sqrt(m["Q"] / m["df"]))


def test_the_additive_model_moves_the_estimate():
    by = [0.05, 0.30, 0.02, 0.40]
    f = M.ivw(BX, SX, by, SY, model="fixed")
    a = M.ivw(BX, SX, by, SY, model="additive")
    assert a["tau2"] > 0.0
    assert abs(a["estimate"] - f["estimate"]) > 1e-6


def test_second_order_weights_differ_from_first_order():
    a = M.ivw(BX, SX, BY, SY, weights="first_order")
    b = M.ivw(BX, SX, BY, SY, weights="second_order")
    assert abs(a["estimate"] - b["estimate"]) > 1e-8


@pytest.mark.parametrize("call", [
    lambda: M.ivw([0.0, 0.2], SX[:2], BY[:2], SY[:2]),
    lambda: M.ivw(BX, SX, BY[:2], SY),
    lambda: M.ivw(BX, SX, BY, [0.0] * 4),
    lambda: M.ivw(BX, SX, BY, SY, model="bayesian"),
    lambda: M.ivw(BX, SX, BY, SY, weights="third_order"),
])
def test_invalid_mr_inputs_are_refused(call):
    with pytest.raises(ValueError):
        call()
