"""Emit the Python arm's values for the R parity check."""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/rootcoder/work/morie/src")
from morie.fn import _schab_st as st
from morie.fn._rng import random_normal

cs = lambda h: 2.0 * np.exp(-h / 3.0)
ct = lambda k: 1.5 * np.exp(-k / 2.0)
cov = lambda d, u: st.separable_covariance(d, u, cs, ct)

h = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
k = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
rs_v = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
rt_v = np.array([1.0, 0.7, 0.5, 0.3, 0.1])
pmf = np.zeros((4, 5))
pmf[2, 3] = 0.6
pmf[0, 0] = 0.4
nodes = [0.5, 1.0, 2.0]
weights = [0.25, 0.5, 0.25]

n = 60
rs = np.random.RandomState(19)
coords = rs.uniform(0, 10, size=(n, 2))
times = np.repeat(np.arange(6, dtype=float), 10)
D, K = st.st_lag_matrices(coords, times)
C = cov(D, K)
C[np.diag_indices_from(C)] += 1e-8
z = np.linalg.cholesky(C) @ random_normal(n, seed=2026, stream=1)

emp = st.empirical_st_semivariogram(coords, times, z, n_space_bins=4,
                                    n_time_bins=3)
cond = st.conditional_spatial_semivariogram(coords, times, z, at_time=2.0,
                                            n_bins=3)
wls = st.st_wls_objective(
    emp, lambda hh, kk: st.semivariogram_from_covariance(hh, kk, cov))

npts = 400
rp = np.random.RandomState(23)
pts = rp.uniform(0, 10, size=(npts, 2))
tt = rp.uniform(0, 5, npts)
region = (0.0, 10.0, 0.0, 10.0)
span = (0.0, 5.0)
lam = st.st_intensity(pts, tt, region, span)
marg = st.st_marginal_intensities(pts, tt, region, span, 4, 4)
cst = st.cstr_test(pts, tt, region, span, 3, 3)

gl_nodes, gl_weights = st.gauss_legendre(24)
bessel_x = np.array([0.0, 0.5, 1.0, 2.404825557695773, 5.0])
bessel_z = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
whittle_h = np.array([0.0, 0.5, 1.0, 2.0])
jz_h = np.array([0.5, 1.0, 2.0])
jz_k = np.array([0.0, 0.5, 1.0])
jz, _ = st.jones_zhang_covariance(jz_h, jz_k, sigma2=1.0, theta=1.0, c=1.0,
                                  p=2.0, d=2, n_quad=40)
chi_x = np.array([0.5, 1.0, 3.841458820694124, 10.0])

out = {
    "h": h.tolist(), "k": k.tolist(),
    "product": st.separable_covariance(h, k, cs, ct, "product").tolist(),
    "sum": st.separable_covariance(h, k, cs, ct, "sum").tolist(),
    "product_sum": st.separable_covariance(h, k, cs, ct, "product_sum").tolist(),
    "exp_sep": st.exponential_separable_correlation(h, k, 0.7, 0.3).tolist(),
    "gneiting": st.gneiting_covariance(h, k, sigma2=2.0, a=0.5, c=0.3,
                                       alpha=1.0, beta=0.8, gamma=1.0,
                                       d=2).tolist(),
    "gneiting_t": st.gneiting_with_temporal(h, k, sigma2=2.0, a=0.5, c=0.3,
                                            alpha=1.0, beta=0.5, beta_t=0.4,
                                            gamma=1.0, d=2).tolist(),
    "rs": rs_v.tolist(), "rt": rt_v.tolist(),
    "pm_poisson": st.power_mixture_correlation(rs_v, rt_v, "poisson",
                                               lam=2.0).tolist(),
    "pm_binom": st.power_mixture_correlation(rs_v, rt_v, "binomial", n=4,
                                             pi=0.3).tolist(),
    "pmf": pmf.tolist(),
    "pm_biv": st.bivariate_power_mixture_correlation(rs_v, rt_v, pmf).tolist(),
    "nodes": nodes, "weights": weights,
    "scale_mix": st.scale_mixture_covariance(h, k, cs, ct, nodes,
                                             weights).tolist(),
    "gl_nodes": gl_nodes.tolist(), "gl_weights": gl_weights.tolist(),
    "bessel_x": bessel_x.tolist(), "j0": st.bessel_j0(bessel_x).tolist(),
    "bessel_z": bessel_z.tolist(), "k1": st._bessel_k1(bessel_z).tolist(),
    "whittle_h": whittle_h.tolist(),
    "whittle": st.whittle_spatial_covariance(whittle_h, 3.0, 1.0).tolist(),
    "jz_h": jz_h.tolist(), "jz_k": jz_k.tolist(), "jz": np.asarray(jz).tolist(),
    "gamma_model": st.semivariogram_from_covariance(h, k, cov).tolist(),
    "coords": coords.T.reshape(-1).tolist(),   # column-major for matrix(ncol=2)
    "times": times.tolist(), "z": z.tolist(),
    "emp_counts": emp["counts"].T.reshape(-1).tolist(),
    "emp_gamma": emp["gamma"].T.reshape(-1).tolist(),
    "at_time": 2.0,
    "cond_gamma": cond["gamma"].tolist(),
    "wls": float(wls),
    "pts": pts.T.reshape(-1).tolist(), "tt": tt.tolist(),
    "region": list(region), "span": list(span),
    "intensity": lam["intensity"],
    "marg_s": marg["marginal_spatial"].T.reshape(-1).tolist(),
    "marg_t": marg["marginal_temporal"].tolist(),
    "cstr_index": cst["index_of_dispersion"], "cstr_p": cst["p_value"],
    "chi_x": chi_x.tolist(),
    "chi_sf": [st._chi2_sf(float(x), 1) for x in chi_x],
    "sep_p": st.separability_test(100.0, 103.0)["p_value"],
}
with open("/tmp/py_st_values.json", "w") as fh:
    json.dump(out, fh)
print("wrote /tmp/py_st_values.json with %d entries" % len(out))
