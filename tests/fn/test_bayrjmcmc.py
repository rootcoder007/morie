"""Tests for bayrjmcmc (Green 1995, reversible-jump MCMC)."""

import math

import pytest

from morie.fn.bayrjmcmc import (birth_log_jacobian, birth_split_heights,
                                changepoint_move_probabilities,
                                changepoint_rjmcmc, check_dimension_matching,
                                numeric_log_jacobian, reversible_jump_mcmc,
                                rj_log_acceptance, step_function_loglik)


def _lognorm(x, sd):
    return (-0.5 * math.log(2.0 * math.pi) - math.log(sd)
            - 0.5 * (x / sd) ** 2)


# --------------------------------------------------------------------------
# section 4-3: the birth map and its Jacobian
# --------------------------------------------------------------------------

def test_birth_keeps_the_weighted_geometric_mean():
    s_l, s_star, s_r = 0.3, 0.55, 1.1
    h_j, u = 2.7, 0.32
    hl, hr = birth_split_heights(h_j, u, s_l, s_star, s_r)
    lhs = (s_star - s_l) * math.log(hl) + (s_r - s_star) * math.log(hr)
    assert abs(lhs - (s_r - s_l) * math.log(h_j)) < 1e-12
    assert abs(hr / hl - u / (1.0 - u)) < 1e-12
    # u = 1/2 leaves the height alone
    a, b = birth_split_heights(h_j, 0.5, s_l, s_star, s_r)
    assert abs(a - h_j) < 1e-12 and abs(b - h_j) < 1e-12


def test_birth_jacobian_matches_the_derivative_of_its_own_map():
    s_l, s_star, s_r = 0.0, 0.4, 1.0
    for h_j in (0.5, 2.7, 11.0):
        for u in (0.15, 0.5, 0.83):
            hl, hr = birth_split_heights(h_j, u, s_l, s_star, s_r)
            ana = birth_log_jacobian(h_j, hl, hr)
            num = numeric_log_jacobian(
                lambda z: list(birth_split_heights(z[0], z[1], s_l,
                                                   s_star, s_r)),
                [h_j, u])
            assert abs(ana - num) < 1e-6
            assert abs(ana - math.log((hl + hr) ** 2 / h_j)) < 1e-12


def test_numeric_log_jacobian_refuses_a_non_square_map():
    with pytest.raises(ValueError):
        numeric_log_jacobian(lambda z: [z[0], z[0], z[0]], [1.0, 2.0])


def test_numeric_log_jacobian_on_a_known_determinant():
    # (a, b) -> (2a + b, a + 3b) has determinant 5
    got = numeric_log_jacobian(lambda z: [2 * z[0] + z[1], z[0] + 3 * z[1]],
                               [0.7, -0.2])
    assert abs(got - math.log(5.0)) < 1e-7


# --------------------------------------------------------------------------
# section 3-3: dimension matching
# --------------------------------------------------------------------------

_M2 = {"a": {"dim": 0, "logpost": lambda t: 0.0},
       "b": {"dim": 1, "logpost": lambda t: 0.0}}


def _pair(n_u=1, n_u_rev=0):
    return [
        {"frm": "a", "to": "b", "n_u": n_u, "n_u_rev": n_u_rev,
         "propose": lambda t, uni: [uni()],
         "transform": lambda t, u: (list(u), [])},
        {"frm": "b", "to": "a", "n_u": n_u_rev, "n_u_rev": n_u,
         "propose": lambda t, uni: [],
         "transform": lambda t, u: ([], list(t))},
    ]


def test_dimension_matching_accepts_a_balanced_pair():
    by_pair = check_dimension_matching(_M2, _pair())
    assert set(by_pair) == {("a", "b"), ("b", "a")}


def test_dimension_matching_rejects_an_unbalanced_pair():
    models = {"a": {"dim": 1, "logpost": lambda t: 0.0},
              "b": {"dim": 3, "logpost": lambda t: 0.0}}
    moves = [
        {"frm": "a", "to": "b", "n_u": 1, "n_u_rev": 0,
         "propose": lambda t, uni: [uni()],
         "transform": lambda t, u: (list(t) + list(u) + [0.0], [])},
        {"frm": "b", "to": "a", "n_u": 0, "n_u_rev": 1,
         "propose": lambda t, uni: [],
         "transform": lambda t, u: (t[:1], [t[1]])},
    ]
    with pytest.raises(ValueError, match="dimension matching"):
        check_dimension_matching(models, moves)


def test_a_move_without_its_reverse_is_rejected():
    with pytest.raises(ValueError, match="no reverse move"):
        check_dimension_matching(_M2, _pair()[:1])


def test_the_reverse_must_agree_about_the_lengths_of_u():
    moves = _pair()
    moves[1]["n_u_rev"] = 2          # says the forward move draws 2, not 1
    with pytest.raises(ValueError):
        check_dimension_matching(_M2, moves)


def test_a_within_model_move_is_not_a_jump():
    with pytest.raises(ValueError, match="within-model"):
        check_dimension_matching(
            _M2, [{"frm": "a", "to": "a", "n_u": 0, "n_u_rev": 0,
                   "propose": lambda t, uni: [],
                   "transform": lambda t, u: (t, u)}])


# --------------------------------------------------------------------------
# equation 7
# --------------------------------------------------------------------------

def test_equation_7_is_the_sum_of_its_four_ratios():
    got = rj_log_acceptance(-3.0, -1.5, math.log(0.25), math.log(0.5),
                            math.log(0.2), math.log(0.8), math.log(3.0))
    want = ((-1.5) - (-3.0) + math.log(0.5 / 0.25)
            + math.log(0.8 / 0.2) + math.log(3.0))
    assert abs(got - want) < 1e-12


def test_equation_8_drops_the_missing_proposal_density():
    # m1 = 0: no u^(1), so logq_u is 0 and eq. 7 collapses to eq. 8
    got = rj_log_acceptance(-2.0, -2.5, math.log(0.5), math.log(0.5),
                            0.0, math.log(0.3), 0.0)
    assert abs(got - (-0.5 + math.log(0.3))) < 1e-12


# --------------------------------------------------------------------------
# the general engine against a posterior available in closed form
# --------------------------------------------------------------------------

TAU = 1.5
Y = [2.4, -1.1]
LOG_PK = [math.log(0.6), math.log(0.4)]


def _logpost(k):
    def f(theta):
        out = LOG_PK[k]
        for i, yi in enumerate(Y):
            out += _lognorm(yi - (theta[i] if i < k else 0.0), 1.0)
        for v in theta:
            out += _lognorm(v, TAU)
        return out
    return f


MODELS = {"k0": {"dim": 0, "logpost": _logpost(0)},
          "k1": {"dim": 1, "logpost": _logpost(1)}}


def _exact():
    lm = []
    for k in range(2):
        out = LOG_PK[k]
        for i, yi in enumerate(Y):
            out += _lognorm(yi, math.sqrt(1.0 + TAU ** 2) if i < k else 1.0)
        lm.append(out)
    mx = max(lm)
    z = sum(math.exp(v - mx) for v in lm)
    return [math.exp(v - mx) / z for v in lm]


def _prior_moves():
    from morie.fn._rng import normal_quantile
    return [
        {"frm": "k0", "to": "k1", "n_u": 1, "n_u_rev": 0,
         "propose": lambda t, uni: [TAU * float(normal_quantile(uni()))],
         "transform": lambda t, u: (list(u), []),
         "logq": lambda t, u: _lognorm(u[0], TAU)},
        {"frm": "k1", "to": "k0", "n_u": 0, "n_u_rev": 1,
         "propose": lambda t, uni: [],
         "transform": lambda t, u: ([], [t[0]]),
         "logq_rev": lambda t2, u2: _lognorm(u2[0], TAU)},
    ]


def test_visited_model_frequencies_match_the_exact_posterior():
    exact = _exact()
    res = reversible_jump_mcmc(MODELS, _prior_moves(), "k0", (),
                               n_iter=120000, burn_in=10000, seed=3,
                               keep_chain=False)
    assert abs(res["model_freq"]["k0"] - exact[0]) < 0.012
    assert abs(res["model_freq"]["k1"] - exact[1]) < 0.012
    assert res["n_kept"] == 110000


def test_a_deleted_jacobian_breaks_it():
    """The Jacobian is load bearing, so the check above is not vacuous."""
    from morie.fn._rng import normal_quantile
    exact = _exact()
    moves = [
        {"frm": "k0", "to": "k1", "n_u": 1, "n_u_rev": 0,
         "propose": lambda t, uni: [uni()],
         "transform": lambda t, u: ([TAU * float(normal_quantile(u[0]))], []),
         "logjac": lambda t, u, t2, u2: 0.0},
        {"frm": "k1", "to": "k0", "n_u": 0, "n_u_rev": 1,
         "propose": lambda t, uni: [],
         "transform": lambda t, u: (
             [], [0.5 * (1.0 + math.erf(t[0] / TAU / math.sqrt(2.0)))]),
         "logjac": lambda t, u, t2, u2: 0.0},
    ]
    res = reversible_jump_mcmc(MODELS, moves, "k0", (), n_iter=60000,
                               burn_in=5000, seed=3, keep_chain=False)
    assert abs(res["model_freq"]["k0"] - exact[0]) > 0.05


def test_the_numeric_jacobian_route_reproduces_the_analytic_one():
    from morie.fn._rng import normal_quantile
    exact = _exact()
    moves = [
        {"frm": "k0", "to": "k1", "n_u": 1, "n_u_rev": 0,
         "propose": lambda t, uni: [uni()],
         "transform": lambda t, u: ([TAU * float(normal_quantile(u[0]))], [])},
        {"frm": "k1", "to": "k0", "n_u": 0, "n_u_rev": 1,
         "propose": lambda t, uni: [],
         "transform": lambda t, u: (
             [], [0.5 * (1.0 + math.erf(t[0] / TAU / math.sqrt(2.0)))])},
    ]
    res = reversible_jump_mcmc(MODELS, moves, "k0", (), n_iter=80000,
                               burn_in=5000, seed=3, jacobian="numeric",
                               keep_chain=False)
    assert abs(res["model_freq"]["k0"] - exact[0]) < 0.02


def test_engine_argument_validation():
    with pytest.raises(ValueError, match="jacobian"):
        reversible_jump_mcmc(MODELS, _prior_moves(), "k0", (),
                             jacobian="magic")
    with pytest.raises(ValueError, match="init_model"):
        reversible_jump_mcmc(MODELS, _prior_moves(), "k9", ())
    with pytest.raises(ValueError, match="init_theta"):
        reversible_jump_mcmc(MODELS, _prior_moves(), "k0", (1.0, 2.0))
    with pytest.raises(ValueError, match="burn_in"):
        reversible_jump_mcmc(MODELS, _prior_moves(), "k0", (), n_iter=10,
                             burn_in=10)


# --------------------------------------------------------------------------
# section 4
# --------------------------------------------------------------------------

def test_equation_9_on_a_hand_computable_step_function():
    # x(t) = 2 on [0, 1), 5 on [1, 3); points at 0.5, 1.5, 2.5
    got = step_function_loglik([0.5, 1.5, 2.5], [1.0], [2.0, 5.0], 3.0)
    want = math.log(2.0) + 2 * math.log(5.0) - (2.0 * 1.0 + 5.0 * 2.0)
    assert abs(got - want) < 1e-12


def test_equation_9_rejects_bad_input():
    with pytest.raises(ValueError):
        step_function_loglik([0.5], [1.0], [2.0], 3.0)       # too few heights
    with pytest.raises(ValueError):
        step_function_loglik([9.0], [1.0], [2.0, 5.0], 3.0)  # point outside
    assert step_function_loglik([0.5], [], [-1.0], 3.0) == float("-inf")


def test_move_probabilities_satisfy_the_section_4_3_conditions():
    eta, pi_, b, d, c = changepoint_move_probabilities(3.0, 30)
    assert d[0] == 0.0 and pi_[0] == 0.0
    assert b[30] == 0.0
    for k in range(31):
        assert abs(eta[k] + pi_[k] + b[k] + d[k] - 1.0) < 1e-12
        assert b[k] + d[k] <= 0.9 + 1e-12
        if k:
            assert abs(eta[k] - pi_[k]) < 1e-12
    # c is as large as the cap allows
    assert abs(max(b[k] + d[k] for k in range(31)) - 0.9) < 1e-12
    # b_k p(k) == d_{k+1} p(k+1)
    for k in range(6):
        pk = math.exp(-3.0 + k * math.log(3.0) - math.lgamma(k + 1.0))
        pk1 = math.exp(-3.0 + (k + 1) * math.log(3.0)
                       - math.lgamma(k + 2.0))
        assert abs(b[k] * pk - d[k + 1] * pk1) < 1e-15


def test_changepoint_argument_validation():
    with pytest.raises(ValueError, match="L must be positive"):
        changepoint_rjmcmc(y=(), L=0.0)
    with pytest.raises(ValueError, match="outside"):
        changepoint_rjmcmc(y=(2.0,), L=1.0)
    with pytest.raises(ValueError, match="alpha and beta"):
        changepoint_rjmcmc(y=(), L=1.0, alpha=0.0)
    with pytest.raises(ValueError, match="k_init"):
        changepoint_rjmcmc(y=(), L=1.0, k_max=4, k_init=9)


def test_without_the_likelihood_the_chain_returns_the_prior():
    lam, k_max = 3.0, 8
    res = changepoint_rjmcmc(y=(), L=40.0, n_iter=200000, burn_in=10000,
                             lam=lam, k_max=k_max, alpha=2.0, beta=5.0,
                             seed=7, use_likelihood=False)
    pk = [math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1.0))
          for k in range(k_max + 1)]
    tot = sum(pk)
    pk = [v / tot for v in pk]
    for k in range(k_max + 1):
        assert abs(res["k_posterior"][k] - pk[k]) < 0.012
    assert abs(res["mean_height"] - 2.0 / 5.0) < 0.03
    # s_1 given k = 1 is the median of three uniforms on [0, L]
    assert abs(res["mean_s1_given_k1"] - 20.0) < 0.6
    assert abs(res["var_s1_given_k1"] - 40.0 ** 2 / 20.0) < 8.0
