"""Emit the Python arm's GLMM values for the R parity check."""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/rootcoder/work/morie/src")
from morie.fn import _schab_glmm as gm
from morie.fn._rng import random_normal

rs = np.random.RandomState(29)
n = 24
X = np.column_stack([np.ones(n), rs.uniform(-1, 1, n)])
beta = np.array([0.45, 0.75])
t = np.linspace(0, 8, n)
d = np.abs(np.subtract.outer(t, t))
Sigma_S = 0.4 * np.exp(-d / 2.5)
S = np.linalg.cholesky(Sigma_S + 1e-10 * np.eye(n)) @ random_normal(n, seed=99, stream=1)
mu = np.exp(X @ beta + S)
z = np.floor(mu + 0.5)                    # deterministic, so both arms match

fit = gm.fit_pseudo_likelihood(z, X, Sigma_S, family="poisson")
sc = gm.pql_score(z, X, fit["beta"], fit["S"], Sigma_S, "poisson", "log")

# areas
nb = 10
A = np.zeros((nb, nb))
for i in range(nb - 1):
    A[i, i + 1] = A[i + 1, i] = 1.0
A[0, 4] = A[4, 0] = 1.0
R = gm.neighbour_structure(A)
E = np.linspace(8.0, 30.0, nb)
Y = np.floor(E * np.exp(np.linspace(-0.3, 0.3, nb)) + 0.5)
u = np.linspace(-0.5, 0.5, nb)
v = np.linspace(0.1, -0.1, nb)

bym = gm.bym_map(Y, E, A, 0.129, 0.011)
Rt1 = gm.random_walk_structure(6, 1)
Rt2 = gm.random_walk_structure(6, 2)
i4 = gm.interaction_structure(R, Rt1, "IV")
c4 = gm.null_space_constraints(i4["structure"])
mom = gm.marginal_moments_lognormal(X, beta, 0.4, sigma2=1.0, rho=np.exp(-d / 2.5))
pred = gm.predict_glm(1.15, 0.08, 2.5, "log")

out = {
    "X": X.T.reshape(-1).tolist(), "beta": beta.tolist(), "S": S.tolist(),
    "z": z.tolist(), "Sigma_S": Sigma_S.T.reshape(-1).tolist(),
    "n": n,
    "cond_mean": gm.conditional_mean(X, beta, S, "log").tolist(),
    "cond_var": gm.conditional_variance(mu, 2.0, "poisson").tolist(),
    "naive": gm.naive_marginal_mean(X, beta, "log").tolist(),
    "marg_mean": mom["mean"].tolist(), "marg_var": mom["variance"].tolist(),
    "marg_cov": mom["covariance"].T.reshape(-1).tolist(),
    "pseudo": gm.pseudo_data(z, mu, "log").tolist(),
    "sigma_mu": gm.sigma_mu(mu, 1.0, "poisson", "log").T.reshape(-1).tolist(),
    "data_cov": gm.data_covariance(mu, 1.0, "poisson").T.reshape(-1).tolist(),
    "fit_beta": fit["beta"].tolist(), "fit_S": fit["S"].tolist(),
    "fit_mu": fit["mu"].tolist(), "fit_se": fit["se_beta"].tolist(),
    "fit_iter": fit["n_iter"],
    "reml": gm.reml_objective(X, fit["Sigma_nu"], fit["pseudo_data"]),
    "score_beta": sc["score_beta"].tolist(), "score_S": sc["score_S"].tolist(),
    "pred": [pred["prediction"], pred["mspe"], pred["inverse_link_prediction"]],
    "A": A.T.reshape(-1).tolist(), "nb": nb,
    "R": R.T.reshape(-1).tolist(),
    "icar_cov": gm.icar_covariance(R).T.reshape(-1).tolist(),
    "icar_fc_mean": gm.icar_full_conditional(u, A)["mean"].tolist(),
    "icar_fc_var": gm.icar_full_conditional(u, A)["variance"].tolist(),
    "lcar_Q": gm.lcar_precision(R, 0.6)[0].T.reshape(-1).tolist(),
    "lcar_fc_mean": gm.lcar_full_conditional(u, A, 0.6)["mean"].tolist(),
    "lcar_fc_var": gm.lcar_full_conditional(u, A, 0.6)["variance"].tolist(),
    "E": E.tolist(), "Y": Y.tolist(), "u": u.tolist(), "v": v.tolist(),
    "smr": gm.smr(Y, E).tolist(),
    "bym_u": bym["u"].tolist(), "bym_v": bym["v"].tolist(),
    "bym_x": bym["x"].tolist(), "bym_sum_v": bym["sum_v"],
    "bym_fitted_total": bym["fitted_total"],
    "bym_logpost": bym["log_posterior"],
    "bym_icar_logprior": gm.bym_icar_log_prior(u, A, 0.129),
    "bym_median_logprior": gm.bym_median_log_prior(u, A, 0.129),
    "rw1": Rt1.T.reshape(-1).tolist(), "rw2": Rt2.T.reshape(-1).tolist(),
    "i4_rank": i4["rank"], "i4_def": i4["rank_deficiency"],
    "c4_n": c4["n_constraints"],
    "sum_to_zero": gm.apply_sum_to_zero(np.arange(nb * 6, dtype=float),
                                        c4["A"]).tolist(),
    "trend": gm.linear_trend_log_risk(0.1, u, 0.05, v,
                                      np.arange(6.0)).T.reshape(-1).tolist(),
    "nonpar": gm.nonparametric_log_risk(
        0.1, u, np.linspace(0, 0.5, 6),
        np.arange(6.0)).T.reshape(-1).tolist(),
}
with open("/tmp/py_glmm_values.json", "w") as fh:
    json.dump(out, fh)
print("wrote /tmp/py_glmm_values.json with %d entries" % len(out))
