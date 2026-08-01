"""Emit the Python arm's values for the R parity check of the last five
Schabenberger modules."""

import json
import sys

import numpy as np

sys.path.insert(0, "/home/rootcoder/work/morie/src")

from morie.fn import _schab_moran as mo
from morie.fn import _schab_nonstat as ns
from morie.fn import _schab_pp as pp
from morie.fn import _schab_spectral as sp


def rook(g):
    n = g * g
    w = np.zeros((n, n))
    for i in range(g):
        for j in range(g):
            k = i * g + j
            for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                a, b = i + di, j + dj
                if 0 <= a < g and 0 <= b < g:
                    w[k, a * g + b] = 1.0
    return w


out = {}

# --- Moran, on the Example 1.7 design ---------------------------------------
g = 10
W = rook(g)
rs = np.random.RandomState(0)
x, y = np.meshgrid(np.arange(g), np.arange(g), indexing="ij")
z = (1.4 + 0.1 * x + 0.2 * y + 0.002 * x ** 2).ravel() + rs.standard_normal(100)
m = mo.moran_moments(z, W)
out["moran_z"] = z.tolist()
out["moran"] = {k: m[k] for k in
                ("I", "expectation", "variance_normal",
                 "variance_randomization", "sd_normal", "sd_randomization",
                 "z_normal", "z_randomization", "kurtosis_b", "S0", "S1",
                 "S2", "geary_c")}

# --- cross-K -----------------------------------------------------------------
reg = (0.0, 0.0, 1.0, 1.0)
r = np.linspace(0.02, 0.25, 8)
p1 = rs.uniform(0, 1, (80, 2))
p2 = rs.uniform(0, 1, (70, 2))
ck = pp.cross_k_combined(p1, p2, reg, r)
dd = pp.diggle_chetwynd_d(p1, p2, reg, r)
wp = rs.uniform(0.02, 0.98, (40, 2))
wt = rs.uniform(0.01, 0.7, 40)
out["pp"] = {
    "p1": p1.T.reshape(-1).tolist(), "p2": p2.T.reshape(-1).tolist(),
    "r": r.tolist(),
    "K_star": ck["K_star"].tolist(), "K_12": ck["K_12"].tolist(),
    "K_21": ck["K_21"].tolist(), "L_minus_h": ck["L_minus_h"].tolist(),
    "D": dd["D"].tolist(), "K_11": dd["K_11"].tolist(),
    "K_22": dd["K_22"].tolist(),
    "wp": wp.T.reshape(-1).tolist(), "wt": wt.tolist(),
    "weights": pp.ripley_weights(wp, reg, wt).tolist(),
    "K12_uncorrected": pp.cross_k_function(p1, p2, reg, r, "none").tolist(),
}

# --- periodogram --------------------------------------------------------------
zz = rs.standard_normal((7, 5)) + 0.4 * np.add.outer(np.arange(7), np.arange(5))
P = sp.periodogram(zz)
Q = sp.periodogram_from_covariance(zz)
cov, lj, lk = sp.sample_covariance_2d(zz)
out["spec"] = {
    "z": zz.T.reshape(-1).tolist(), "r": 7, "c": 5,
    "periodogram": P["periodogram"].T.reshape(-1).tolist(),
    "from_cov": Q["periodogram"].T.reshape(-1).tolist(),
    "omega1": P["omega1"].tolist(), "omega2": P["omega2"].tolist(),
    "cov": cov.T.reshape(-1).tolist(),
    "mean_invariant": bool(P["mean_invariant"]),
}

# --- point source + moving windows --------------------------------------------
s = rs.uniform(0, 10, (36, 2))
src = np.array([4.0, 6.0])
psc = ns.point_source_correlation(s, src, 0.35, 0.12, 0.07)
zf = np.sin(s[:, 0] / 2.0) + 0.25 * rs.standard_normal(36)
big = rs.uniform(0, 10, (120, 2))
zbig = np.sin(big[:, 0] / 2.0) + 0.25 * rs.standard_normal(120)
tg = np.array([[3.0, 3.0], [7.0, 7.0]])
hw = ns.haas_window(big, tg[0], min_sites=35, step=5)
mw = ns.moving_window_krige(big, zbig, tg, min_sites=35, step=5,
                            local_variogram=True)
lk = ns.moving_window_krige(big, zbig, tg, min_sites=35, step=5,
                            local_variogram=False)
out["ns"] = {
    "s": s.T.reshape(-1).tolist(), "src": src.tolist(),
    "corr": psc["correlation"].T.reshape(-1).tolist(),
    "ci": psc["source_distance"].tolist(),
    "min_eig": psc["min_eigenvalue"],
    "big": big.T.reshape(-1).tolist(), "zbig": zbig.tolist(),
    "tg": tg.T.reshape(-1).tolist(),
    "haas_n": hw["n_sites"], "haas_radius": hw["radius"],
    "haas_counts": hw["lag_counts"].tolist(),
    "haas_index_sorted": sorted(int(i) for i in hw["index"]),
    "mw_pred": mw["prediction"].tolist(),
    "mw_sill": mw["local_sill"].tolist(),
    "mw_range": mw["local_range"].tolist(),
    "mw_sizes": mw["window_sizes"].tolist(),
    "lk_pred": lk["prediction"].tolist(),
    "lk_range": lk["local_range"].tolist(),
    "global_sill": mw["global_sill"], "global_range": mw["global_range"],
    "pr": ns.practical_range(0.35),
    "pr_pair": float(ns.practical_range(0.35, 0.12, 0.07, ci=2.0, cj=3.5)),
}

with open("/tmp/py_rest_values.json", "w") as fh:
    json.dump(out, fh)
print("wrote /tmp/py_rest_values.json")
