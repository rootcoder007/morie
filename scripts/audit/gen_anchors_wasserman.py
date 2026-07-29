"""Generate R-parity anchors for the Wasserman shelf from the live Python core."""
import importlib, json, sys
sys.path.insert(0, "src")
A = {}
def payload(name, fn, *args, keys=None, **kw):
    mod = importlib.import_module(f"morie.fn.{name}")
    assert "/src/morie/fn/" in mod.__file__, mod.__file__
    out = getattr(mod, fn)(*args, **kw)
    d = dict(out)
    if keys: d = {k: d[k] for k in keys}
    A[name] = d

payload("wsmvar", "wasserman_variance", [1.0, 2.0, 3.0, 4.0])
payload("wsmcby", "wasserman_chebyshev_ineq", [2.0, 0.5])
payload("wsmcdf", "wasserman_empirical_cdf", [2.5, 2.0, 0.0], [1.0, 2.0, 3.0, 4.0])
payload("wsmexp", "wasserman_expectation", [i / 1000.0 for i in range(1001)], [1.0] * 1001)
payload("wsmcov", "wasserman_covariance", [1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
payload("wsmmrk", "wasserman_markov_ineq", 3.0, 2.0)
payload("wsmhfd", "wasserman_hoeffding", 100, 0.1, 0.0, 1.0)
payload("wsmmgf", "wasserman_mgf", [0.0, 1.0, 2.0], [0.0, 0.5, 1.0])
payload("wsmcfn", "wasserman_char_fn", [0.3, -2.0, 5.5], [0.5, 1.5])
payload("wsmclt", "wasserman_clt", [1.0, 2.0, 3.0, 4.0])
payload("wsmlln", "wasserman_lln", [2.0, 4.0, 6.0, 1.0])
payload("wsmdlm", "wasserman_delta_method", 3.0, 0.5, 6.0)
payload("wsmqtl", "wasserman_empirical_quantile", [3.0, 1.0, 4.0, 2.0], [0.25, 0.5, 0.51, 1.0])
payload("wsmcb", "wasserman_dkw_cb", [1.0, 2.0, 5.0], 0.05)
payload("wsmnpb", "wasserman_nonparametric_boot", [1.0, 2.0, 3.0, 4.0], None, 50)
payload("wsmbpc", "wasserman_bootstrap_percentile", [1.0, 2.0, 3.0, 4.0], None, 50, 0.10)
payload("wsmbpv", "wasserman_bootstrap_pivotal", [1.0, 2.0, 3.0, 4.0], None, 50, 0.10)
payload("wsmprb", "wasserman_parametric_boot", [1.0, 2.0, 3.0, 4.0], None, None, 50)
payload("wsmifn", "wasserman_influence_function", [1.0, 2.0, 3.0, 4.0], None)
payload("wsmlik", "wasserman_likelihood", [1.0, 2.0], None, 2.0)
payload("wsmllk", "wasserman_log_likelihood", [1.0, 2.0], None, 2.0)
payload("wsmcrl", "wasserman_cramer_rao", 0.0, 25, 0.25)
payload("wsmfis", "wasserman_fisher_info", None, 2.0)
payload("wsmasm", "wasserman_mle_asymptotic", list(range(100)), None, 2.0)
import numpy as np
X = np.array([[1.0, -1.0], [1.0, 1.0]] * 50); e = np.tile([0.5, -0.5, -0.5, 0.5], 25)
payload("wsmwhz", "wasserman_white_huber", X, X @ np.array([1.0, 2.0]) + e)
payload("wsmemt", "wasserman_em_algorithm", [0.0, 0.1, -0.1, 0.05, 10.0, 10.1, 9.9, 10.05], (0.5, -1.0, 11.0, 1.0, 1.0))
payload("wsmchi", "wasserman_chi_sq_gof", [10, 20, 30], [20, 20, 20])
g = np.linspace(-5.0, 5.0, 2001)
payload("wsmbay", "wasserman_posterior", [1.0, 0.5, 1.5], None, (g, np.ones_like(g)), keys=["estimate", "evidence", "map_theta", "n"])
g2 = np.linspace(0.0, 1.0, 10001)
payload("wsmbcr", "wasserman_credible_interval", (g2, np.ones_like(g2)), 0.10)
payload("wsmpst1", "wasserman_posterior_mean", (g2, np.ones_like(g2)))
payload("bfsd", "bayes_factor_savage_dickey", (np.linspace(-3, 3, 2001) / 3.0), 1.0 / 6.0, 0.0)
payload("wsment", "wasserman_entropy", [0.5, 0.25, 0.25])
payload("wsmkbk", "wasserman_kullback_leibler", [0.5, 0.5], [0.25, 0.75])
payload("wsmmtl", "wasserman_mutual_info", [0, 0, 1, 1], [0, 0, 1, 1])
payload("wsmodd", "wasserman_odds_ratio", [[30, 10], [15, 45]])
payload("wsmrrr", "wasserman_relative_risk", [[30, 70], [10, 90]])
payload("densty", "density", [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
payload("sgtclo", "sgt_closeness_centrality", [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
payload("wsmlsr", "wasserman_least_squares", [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 4.0]], [1.0, 3.2, 4.9, 9.1])
payload("wsmrgr", "wasserman_ridge", [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], [1.0, 3.0, 5.0], 0.5)
payload("wsmlas", "wasserman_lasso", [[1.0, 0.0], [0.0, 1.0]], [3.0, -1.0], 0.5)
payload("wsmlgr", "wasserman_logistic_regression", [[1.0]] * 4, [1, 1, 1, 0])
payload("wsmpsr", "wasserman_poisson_regression", [[1.0]] * 4, [1, 2, 3, 2])
payload("wsmaic", "wasserman_aic", -100.0, 3)
payload("wsmbic", "wasserman_bic", -100.0, 3, 50)
payload("wsmcvr", "wasserman_kfold_cv", [[1.0, float(i)] for i in range(8)], [1.0 + 2.0 * i + (0.1 if i % 2 else -0.1) for i in range(8)], None, 4)
payload("wsmmcd", "wasserman_mcdiarmid", 0.1, [0.01] * 100)
payload("wsmkbw", "wasserman_kde_bandwidth", list(range(1, 101)))
payload("wsmwlt", "wasserman_wavelet_smooth", [10.0, -10.0, 10.0, -10.0], sigma=1.0)
payload("wsmcrk", "wasserman_kernel_regression", [1.5, 0.0], [0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0], 0.7)
payload("wsmlpr", "wasserman_local_polynomial", [1.5], [0.0, 1.0, 2.0, 3.0], [1.0, 3.1, 4.8, 7.2], 0.7, 1)
payload("wsmsmp", "wasserman_smoothing_spline", [0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 0.0, 2.0], 1.0)
payload("wsmpca", "wasserman_pca", [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.5]], 2)
payload("wsmkmn", "wasserman_kmeans", [[0.0], [0.2], [-0.2], [10.0], [10.2], [9.8]], 2)
payload("wsmgmm", "wasserman_gmm_em", [0.0, 0.1, -0.1, 0.05, 10.0, 10.1, 9.9, 10.05], 2)
payload("wsmhmm", "wasserman_hmm_forward", [0, 1, 0], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0.6, 0.4])
payload("wsmvit", "wasserman_viterbi", [0, 1, 0], [[0.7, 0.3], [0.4, 0.6]], [[0.9, 0.1], [0.2, 0.8]], [0.6, 0.4])
payload("wsmgib", "wasserman_gibbs_sampler", 0.9, [0.0, 0.0], 200, keys=["estimate", "mean_x", "mean_y", "n"])
import math
payload("wsmmcm", "wasserman_mcmc_metropolis", lambda x: math.exp(-0.5 * x * x), 1.0, 0.0, 200, keys=["estimate", "acceptance_rate", "n"])
payload("wsmdir", "wasserman_directed_graph", [{"parents": [], "cpt": {(): 0.3}}, {"parents": [0], "cpt": {(0,): 0.2, (1,): 0.9}}], [1, 1])
agree = lambda t: 2.0 if t[0] == t[1] else 1.0
payload("wsmund", "wasserman_undirected_graph", (2, [(0, 1)]), [agree])
payload("wsmgrp", "wasserman_graphical_model", (2, [(0, 1)]), [agree])
payload("wsmlgc", "wasserman_log_linear", [[30, 10], [15, 45]])
payload("wsmbst", "wasserman_boosting", [[0.0], [1.0], [2.0], [3.0]], [1, -1, 1, -1], None, 5)
payload("wsmsvm", "wasserman_svm", [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]], [-1, -1, 1, 1])
payload("wsmmin", "wasserman_minimax", [[1.0, 4.0], [2.0, 2.0], [3.0, 1.0]], ["T1", "T2", "T3"], ["F1", "F2"])

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, float) and (o != o): return "NaN"
    if o == float("inf"): return "Inf"
    if o == float("-inf"): return "-Inf"
    return o
json.dump(clean(A), open("scripts/audit/anchors_wasserman.json", "w"), indent=1)
print("anchors:", len(A))
