"""Emit the Python arm's GWR values for the R parity check."""

import json
import sys

import numpy as np

sys.path.insert(0, "/home/rootcoder/work/morie/src")

from morie.fn import _schab_gwr as g  # noqa: E402

rs = np.random.RandomState(41)
n = 26
coords = np.column_stack([rs.uniform(0, 10, n), rs.uniform(0, 10, n)])
X = np.column_stack([np.ones(n), rs.uniform(-1, 1, n), rs.uniform(0, 2, n)])
beta = 1.0 + 0.8 * np.sin(0.9 * coords[:, 0])
y = X[:, 0] + beta * X[:, 1] - 0.5 * X[:, 2] + 0.05 * rs.standard_normal(n)
D = g.pairwise_distances(coords)
BW = 2.5

d_vec = np.linspace(0.0, 2.0 * BW, 17)
fit = g.gwr_fit(y, X, D, BW)
sel_cv = g.select_bandwidth(y, X, coords, criterion="cv")
sel_aicc = g.select_bandwidth(y, X, coords, criterion="aicc")
sel_ad = g.select_bandwidth(y, X, coords, criterion="aicc", adaptive=True)

# Two-scale fixture for MGWR.
Xm = np.column_stack([np.ones(n), rs.uniform(-1, 1, n)])
b0 = 1.0 + 0.05 * coords[:, 0]
b1 = np.sin(coords[:, 0])
ym = b0 + b1 * Xm[:, 1] + 0.05 * rs.standard_normal(n)
mg = g.mgwr_backfit(ym, Xm, coords, criterion="aicc", tol=1e-4, max_iter=25)

out = {
    "n": n,
    "coords": coords.T.reshape(-1).tolist(),
    "X": X.T.reshape(-1).tolist(),
    "y": y.tolist(),
    "bw": BW,
    "D": D.T.reshape(-1).tolist(),
    "d_vec": d_vec.tolist(),
    "w_gaussian": g.kernel_weights(d_vec, BW, "gaussian").tolist(),
    "w_gaussian_norm": g.kernel_weights(d_vec, BW, "gaussian", normalized=True).tolist(),
    "w_bisquare": g.kernel_weights(d_vec, BW, "bisquare").tolist(),
    "w_tricube": g.kernel_weights(d_vec, BW, "tricube").tolist(),
    "w_boxcar": g.kernel_weights(d_vec, BW, "boxcar").tolist(),
    "adaptive_bw": [g.adaptive_bandwidth(D[0], k) for k in (3, 8, 15)],
    "params": fit["params"].T.reshape(-1).tolist(),
    "S": fit["S"].T.reshape(-1).tolist(),
    "fitted": fit["fitted"].tolist(),
    "resid": fit["resid"].tolist(),
    "tr_S": fit["tr_S"],
    "tr_STS": fit["tr_STS"],
    "enp": fit["effective_parameters"],
    "rss": fit["rss"],
    "sigma2": fit["sigma2"],
    "sigma2_cressie": fit["sigma2_cressie"],
    "n_rank_deficient": fit["n_rank_deficient"],
    "sigma2_gwr": fit["sigma2_gwr"],
    "edf_resid": fit["edf_resid"],
    "se_params": fit["se_params"].T.reshape(-1).tolist(),
    "aicc": g.aicc_from_parts(n, fit["sigma2"], fit["tr_S"]),
    "aic": g.aic_from_parts(n, fit["sigma2"], fit["tr_S"]),
    "cv": g.cv_score(y, X, D, BW),
    "cv_bisquare": g.cv_score(y, X, D, BW, "bisquare"),
    "sel_cv": sel_cv["bandwidth"],
    "sel_cv_score": sel_cv["score"],
    "sel_aicc": sel_aicc["bandwidth"],
    "sel_aicc_score": sel_aicc["score"],
    "sel_bounds": list(sel_cv["bounds"]),
    "sel_adaptive": sel_ad["bandwidth"],
    "golden_parabola": list(g.golden_section(lambda t: (t - 2.75) ** 2 + 1.0, 0.0, 10.0, tol=1e-9)),
    "Xm": Xm.T.reshape(-1).tolist(),
    "ym": ym.tolist(),
    "mgwr_bws": mg["bandwidths"].tolist(),
    "mgwr_params": mg["params"].T.reshape(-1).tolist(),
    "mgwr_fitted": mg["fitted"].tolist(),
    "mgwr_bw_gwr": mg["bandwidth_gwr"],
    "mgwr_n_iter": mg["n_iter"],
    "mgwr_scores": mg["score_history"],
    "mgwr_converged": mg["converged"],
    "mgwr_boundary": mg["at_search_boundary"],
    "mgwr_standardized": mg["standardized"],
    "mgwr_y_scale": mg["y_scale"],
    "mgwr_x_scale": mg["x_scale"].tolist(),
    # published spgwr NY8 output
    "ny8": [281, 119.6, 568.0, 561.6],
}
with open("/tmp/py_gwr_values.json", "w") as fh:
    json.dump(out, fh)
print("wrote /tmp/py_gwr_values.json with %d entries" % len(out))
