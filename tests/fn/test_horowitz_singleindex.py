"""Single-index estimators: rank, weight function, one-step, discrete."""

from morie.fn import _array_core as np
import pytest

from morie.fn.hrzasym import horowitz_one_step_efficient
from morie.fn.hrzdiscd import horowitz_direct_discrete_x
from morie.fn.hrzrank import horowitz_semipar_rank
from morie.fn.hrzwfun import horowitz_nls_weight_function


def _index(n=400, seed=0, hetero=False):
    """Y = G(X'beta) + noise with G monotone, beta = (1, -0.6)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
    beta = np.array([1.0, -0.6])
    idx = X @ beta
    sd = (0.3 * (1.0 + np.abs(X[:, 0]))) if hetero else np.full(n, 0.3)
    y = np.tanh(idx) + rng.standard_normal(n) * sd
    return X, y, beta


def test_rank_estimator_recovers_the_index_without_any_bandwidth():
    X, y, beta = _index(300)
    out = horowitz_semipar_rank(X, y)
    assert abs(out["beta"][1] - beta[1]) < 0.2
    # the whole appeal: no smoothing parameter anywhere
    assert out["requires_bandwidth"] is False
    # and the whole cost: not efficient, and no analytic SE offered
    assert out["asymptotically_efficient"] is False
    assert out["se"] is None
    assert out["inference"] == "bootstrap"


def test_rank_estimator_only_uses_the_ordering_of_y():
    X, y, _ = _index(250, seed=4)
    a = horowitz_semipar_rank(X, y)
    # any strictly increasing transform of y must give the same answer
    b = horowitz_semipar_rank(X, np.exp(y / 2.0))
    assert np.allclose(a["beta"], b["beta"])
    c = horowitz_semipar_rank(X, y, variant="cs")
    assert abs(c["beta"][1] - a["beta"][1]) < 0.3
    assert c["variant"] == "cs"
    with pytest.raises(ValueError):
        horowitz_semipar_rank(X, y, variant="spearman")


def test_efficient_weights_are_the_reciprocal_of_the_variance_function():
    X, y, beta = _index(600, hetero=True)
    out = horowitz_nls_weight_function(X, y, beta_hat=beta)
    assert out["efficient_weight_used"] is True
    # sigma^2 rises with |x1| by construction, so the weights must fall
    lo = np.abs(X[:, 0]) < 0.5
    hi = np.abs(X[:, 0]) > 1.5
    assert np.median(out["sigma2_hat"][hi]) > np.median(out["sigma2_hat"][lo])
    # compare MEDIANS: the mean of 1/s^2 is dominated by its smallest
    # denominators, so a handful of edge observations where the
    # variance regression dips can invert the mean ordering while the
    # weight function is behaving exactly as intended. That the
    # largest weight is far above the typical one is the estimator's
    # real fragility, and it is reported rather than hidden.
    assert np.median(out["weights"][hi]) < np.median(out["weights"][lo])
    assert out["max_weight"] > 5 * np.median(out["weights"])
    assert np.allclose(out["weights"], 1.0 / out["sigma2_hat"])
    # unknown G costs efficiency but NOT rate; the two are separate
    assert out["efficiency_loss_from_unknown_G"] is True
    assert out["rate_loss_from_unknown_G"] is False


def test_the_sandwich_reaches_the_bound_only_under_efficient_weighting():
    X, y, beta = _index(2000, hetero=True)
    eff = horowitz_nls_weight_function(X, y, beta_hat=beta)
    ineff = horowitz_nls_weight_function(X, y, beta_hat=beta,
                                         weights=np.ones(X.shape[0]))
    # Omega_n = C^{-1} D C^{-1} always. Under W = 1/sigma^2 it
    # converges to the bound Omega_SI; under a flat weight it does
    # not, and cannot be below it.
    d_eff = abs(eff["omega"][0, 0] / eff["omega_SI"][0, 0] - 1.0)
    d_in = abs(ineff["omega"][0, 0] / ineff["omega_SI"][0, 0] - 1.0)
    assert d_eff < d_in
    assert eff["efficient_weight_used"] is True
    assert ineff["efficient_weight_used"] is False
    with pytest.raises(ValueError):
        horowitz_nls_weight_function(X, y, weights=-np.ones(X.shape[0]))


def test_one_step_moves_toward_the_truth_and_needs_only_one_step():
    X, y, beta = _index(500, seed=2)
    start = np.array([1.0, -0.2])          # deliberately off
    out = horowitz_one_step_efficient(X, y, initial_estimator=start)
    assert abs(out["beta"][1] - beta[1]) < abs(start[1] - beta[1])
    assert out["attains_omega_SI"] is True
    assert out["theory_requires_steps"] == 1
    # iterating adds nothing asymptotically: a second step barely moves
    two = horowitz_one_step_efficient(X, y, initial_estimator=start, n_steps=2)
    first_move = abs(out["beta"][1] - start[1])
    second_move = abs(two["beta"][1] - out["beta"][1])
    assert second_move < first_move
    with pytest.raises(ValueError):
        horowitz_one_step_efficient(X, y, initial_estimator=start, n_steps=0)
    with pytest.raises(ValueError):
        horowitz_one_step_efficient(X, y, initial_estimator=[0.0, 1.0])


def test_one_step_defaults_to_a_direct_estimator_as_its_start():
    X, y, beta = _index(400, seed=3)
    out = horowitz_one_step_efficient(X, y)
    # the default start is the average-derivative estimate, which is
    # the intended use: cheap direct estimator, then one step
    assert out["beta_initial"][0] == 1.0
    assert out["se"].size == X.shape[1] - 1


def test_discrete_covariates_need_the_linear_route_not_a_derivative():
    rng = np.random.default_rng(7)
    n = 1200
    X = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
    Z = rng.integers(0, 3, n).astype(float)      # three discrete levels
    beta = np.array([1.0, -0.6])
    alpha = 0.8
    idx = X @ beta + Z * alpha
    y = np.tanh(idx) + rng.standard_normal(n) * 0.2
    out = horowitz_direct_discrete_x(X, y, Z)
    assert out["average_derivative_can_estimate_alpha"] is False
    assert abs(out["beta"][1] - beta[1]) < 0.3
    assert out["identified"] is True
    assert out["support_z"].shape[0] == 3
    assert out["alpha"] is not None
    # a shift in Z shifts G horizontally, so J must move monotonically
    # with z when alpha > 0
    assert out["J"][2] > out["J"][0]


def test_discrete_route_degenerates_gracefully():
    X, y, _ = _index(300, seed=8)
    plain = horowitz_direct_discrete_x(X, y)
    assert plain["alpha"] is None
    assert plain["dz"] == 0
    with pytest.raises(ValueError):      # a single Z value identifies nothing
        horowitz_direct_discrete_x(X, y, np.ones(X.shape[0]))
    with pytest.raises(ValueError):      # strata too small
        horowitz_direct_discrete_x(X, y, np.arange(X.shape[0]) % 100)


def test_discrete_route_accepts_a_supplied_beta_as_well():
    rng = np.random.default_rng(7)
    n = 1200
    X = np.column_stack([rng.standard_normal(n), rng.standard_normal(n)])
    Z = rng.integers(0, 3, n).astype(float)
    beta = np.array([1.0, -0.6])
    y = np.tanh(X @ beta + Z * 0.8) + rng.standard_normal(n) * 0.2

    est = horowitz_direct_discrete_x(X, y, Z)
    sup = horowitz_direct_discrete_x(X, y, Z, beta=beta)
    # both routes are offered: a caller who already ran one of the
    # Sec. 2.5-2.6 estimators has no reason to re-estimate beta here
    assert est["beta_source"] == "stratum-wise (2.46)"
    assert sup["beta_source"] == "supplied"
    assert np.allclose(sup["beta"], beta)
    assert sup["delta_by_stratum"] is None
    # alpha is what the section is actually about, and both routes
    # recover it
    assert abs(est["alpha"][0] - 0.8) < 0.25
    assert abs(sup["alpha"][0] - 0.8) < 0.25
    # scale normalisation still applies to a supplied beta
    doubled = horowitz_direct_discrete_x(X, y, Z, beta=2 * beta)
    assert np.allclose(doubled["beta"], sup["beta"])
    with pytest.raises(ValueError):
        horowitz_direct_discrete_x(X, y, Z, beta=[0.0, 1.0])
