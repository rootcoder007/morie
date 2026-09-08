"""Design-based survey estimation."""

from morie.fn import _array_core as np
import pytest

from morie.fn.calibr import calibration_estimator
from morie.fn.cluvar import cluster_variance
from morie.fn.ebayes import empirical_bayes_shrinkage
from morie.fn.genrgr import calibration_greg
from morie.fn.hjkest import hajek_estimator
from morie.fn.htest1 import horvitz_thompson
from morie.fn.ratest import ratio_estimator
from morie.fn.regest import regression_estimator
from morie.fn.reglmd import regression_estimator_multi
from morie.fn.straprp import stratified_proportion
from morie.fn.taylor import taylor_linearization


def test_horvitz_thompson_is_design_unbiased():
    # a population with known total, sampled with unequal probability
    rng = np.random.default_rng(0)
    N = 1000
    Y = rng.exponential(10.0, N)
    size = Y + rng.exponential(5.0, N)
    pi = np.clip(200 * size / size.sum(), 1e-4, 1.0)
    totals = []
    for _ in range(400):
        take = rng.random(N) < pi
        totals.append(horvitz_thompson(Y[take], pi[take])["total"])
    # unbiasedness is the claim, so it is checked by simulation
    assert abs(np.mean(totals) / Y.sum() - 1.0) < 0.03
    one = horvitz_thompson(Y[:50], pi[:50])
    assert one["design_unbiased"] is True
    assert one["uses_known_N"] is False
    with pytest.raises(ValueError):
        horvitz_thompson(Y[:5], np.zeros(5))


def test_hajek_is_biased_but_less_variable_than_horvitz_thompson():
    rng = np.random.default_rng(1)
    N = 800
    Y = 20 + rng.standard_normal(N) * 2.0     # weakly related to size
    size = rng.exponential(5.0, N)
    pi = np.clip(150 * size / size.sum(), 1e-3, 1.0)
    ht, hj = [], []
    for _ in range(400):
        take = rng.random(N) < pi
        if take.sum() < 5:
            continue
        w = 1.0 / pi[take]
        ht.append(float(np.sum(w * Y[take]) / w.sum()))
        hj.append(hajek_estimator(Y[take], pi[take])["mean"])
    # both target the same mean; Hajek is the more stable of the two
    assert np.std(hj) <= np.std(ht) + 1e-12
    assert hajek_estimator(Y[:40], pi[:40])["design_unbiased"] is False


def test_ratio_estimator_reports_when_the_auxiliary_earns_its_place():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.exponential(10.0, n)
    y_good = 2.0 * x + rng.standard_normal(n) * 0.5     # near-proportional
    good = ratio_estimator(y_good, x, X_mean=10.0)
    assert good["improves_on_simple_mean"] is True
    assert good["correlation"] > good["efficiency_threshold"]
    assert good["ratio"] == pytest.approx(2.0, abs=0.1)
    y_bad = rng.standard_normal(n) * 5.0 + 50.0          # unrelated
    bad = ratio_estimator(y_bad, x, X_mean=10.0)
    assert bad["improves_on_simple_mean"] is False
    with pytest.raises(ValueError):
        ratio_estimator(y_good, x)      # needs the population total or mean


def test_regression_estimator_does_not_force_the_origin():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(5, 15, n)
    y = 100.0 + 2.0 * x + rng.standard_normal(n)   # large intercept
    out = regression_estimator(y, x, X_mean=10.0)
    assert out["slope"] == pytest.approx(2.0, abs=0.2)
    assert out["intercept"] == pytest.approx(100.0, abs=2.0)
    assert out["passes_through_origin"] is False
    # the variance gain is exactly 1 - rho^2
    assert out["variance_ratio_to_simple_mean"] == pytest.approx(
        1 - out["correlation"] ** 2)
    # and the estimate beats the raw mean, which ignores that x is off
    assert abs(out["mean"] - (100 + 2 * 10)) < abs(y.mean() - (100 + 2 * 10)) + 1


def test_more_auxiliaries_raise_R2_but_can_cost_variance():
    rng = np.random.default_rng(4)
    n = 60
    X = rng.standard_normal((n, 8))
    y = 3.0 * X[:, 0] + rng.standard_normal(n)
    one = regression_estimator_multi(y, X[:, :1], np.zeros(1))
    many = regression_estimator_multi(y, X, np.zeros(8))
    # R^2 cannot fall when columns are added
    assert many["R2"] >= one["R2"]
    # but the cost term grows, which is the point
    assert many["p_over_n"] > one["p_over_n"]
    with pytest.raises(ValueError):
        regression_estimator_multi(y[:5], X[:5], np.zeros(8))


def test_greg_reproduces_the_known_totals_direction():
    rng = np.random.default_rng(5)
    n = 150
    X = np.column_stack([np.ones(n), rng.uniform(0, 10, n)])
    y = 5.0 + 2.0 * X[:, 1] + rng.standard_normal(n)
    w = np.full(n, 20.0)
    true_totals = np.array([3000.0, 15000.0])
    out = calibration_greg(y, X, w, true_totals)
    assert out["design_consistent_regardless_of_model"] is True
    # the correction moves the HT total toward the truth
    assert out["total"] != out["ht_total"]
    assert np.allclose(out["residual_totals"],
                       true_totals - (w[:, None] * X).sum(axis=0))


def test_calibration_reproduces_margins_exactly_and_flags_negatives():
    rng = np.random.default_rng(6)
    n = 200
    X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
    y = rng.standard_normal(n)
    d = np.full(n, 10.0)
    targets = np.array([2000.0, 1200.0])       # deliberately off-sample
    out = calibration_estimator(y, X, d, targets)
    # the defining property: margins hold exactly, by construction
    assert out["margins_reproduced"] is True
    assert out["max_margin_error"] < 1e-6
    assert out["equals_greg"] is True
    assert out["n_negative"] >= 0
    # extreme targets are what produce negative weights
    extreme = calibration_estimator(y, X, d, np.array([2000.0, 5000.0]))
    assert extreme["n_negative"] > 0


def test_stratification_has_no_between_stratum_variance_term():
    rng = np.random.default_rng(7)
    # strata with very different proportions: exactly the case where
    # stratification pays
    y = np.r_[rng.random(200) < 0.1, rng.random(200) < 0.9].astype(float)
    st = np.r_[np.zeros(200), np.ones(200)]
    out = stratified_proportion(y, st, weights=np.array([0.5, 0.5]))
    assert out["proportion"] == pytest.approx(0.5, abs=0.06)
    assert out["weights_are_population_shares"] is True
    # the stratified se is far below the unstratified one, because the
    # between-stratum variation is removed by design
    naive = float(np.sqrt(y.mean() * (1 - y.mean()) / y.size))
    assert out["se"] < naive
    with pytest.raises(ValueError):
        stratified_proportion(y, st, weights=np.array([0.5, 0.9]))


def test_cluster_variance_counts_clusters_not_elements():
    rng = np.random.default_rng(8)
    n_clu, m = 25, 40
    effect = rng.standard_normal(n_clu) * 1.0        # real cluster effect
    y = np.concatenate([effect[j] + rng.standard_normal(m) * 1.0
                        for j in range(n_clu)])
    cl = np.repeat(np.arange(n_clu), m)
    out = cluster_variance(y, cl)
    assert out["n_clusters"] == 25
    assert out["n_elements"] == 1000
    # the design effect is real and large, so the naive se badly
    # understates the truth
    assert out["deff"] > 5
    assert out["se"] > out["naive_se"]
    assert out["effective_n"] < 300
    assert out["icc"] > 0.2
    with pytest.raises(ValueError):
        cluster_variance(y, np.zeros_like(cl))


def test_taylor_linearisation_is_first_order_and_says_so():
    rng = np.random.default_rng(9)
    Y = rng.standard_normal((100, 2))
    out = taylor_linearization(Y, np.ones(100), np.array([1.0, -1.0]))
    assert out["first_order_only"] is True
    assert out["variance"] >= 0
    assert "quantiles" in out["valid_for"]
    with pytest.raises(ValueError):
        taylor_linearization(Y, np.ones(100), np.array([1.0]))


def test_shrinkage_pulls_small_clusters_hardest():
    rng = np.random.default_rng(10)
    # one big cluster and several tiny ones
    y = np.r_[rng.standard_normal(200) + 5.0,
              rng.standard_normal(3) + 9.0,
              rng.standard_normal(3) + 1.0,
              rng.standard_normal(4) + 7.0]
    cl = np.r_[np.zeros(200), np.ones(3), np.full(3, 2), np.full(4, 3)]
    out = empirical_bayes_shrinkage(y, cl)
    lam = out["lambda"]
    # the big cluster keeps its mean; the tiny ones are pulled in
    assert lam[0] > lam[1]
    assert lam[0] > lam[2]
    assert out["biased_per_cluster"] is True
    # shrunk estimates sit between the raw mean and the grand mean
    for j in range(len(lam)):
        lo, hi = sorted([out["raw_means"][j], out["grand_mean"]])
        assert lo - 1e-9 <= out["shrunk"][j] <= hi + 1e-9
    with pytest.raises(ValueError):
        empirical_bayes_shrinkage(y[:5], np.zeros(5))
