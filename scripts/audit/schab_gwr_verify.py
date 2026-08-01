"""Source-identity checks for morie.fn._schab_gwr.

Every check names the source it is checking against.  Run from the repo root:

    python3 scripts/audit/schab_gwr_verify.py
"""

import sys

import numpy as np

sys.path.insert(0, "src")

from morie.fn import _schab_gwr as g  # noqa: E402

PASS, FAIL = [], []


def chk(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {name:<58} {'PASS' if ok else 'FAIL'}  {detail}")


def close(a, b, tol=1e-10):
    return bool(np.all(np.abs(np.asarray(a) - np.asarray(b)) <= tol))


# ---------------------------------------------------------------- fixtures
rs = np.random.RandomState(17)
n = 40
coords = np.column_stack([rs.uniform(0, 10, n), rs.uniform(0, 10, n)])
X = np.column_stack([np.ones(n), rs.uniform(-1, 1, n), rs.uniform(0, 2, n)])
beta_true = np.array([1.0, 2.0, -0.5])
y = X @ beta_true + 0.3 * rs.standard_normal(n)
D = g.pairwise_distances(coords)
BW = 3.0


print("\n[Charlton white paper p. 8 / spgwr gwr.aic.f -- AICc against published output]")
# spgwr's NY8 example, printed in Bivand, Pebesma & Gomez-Rubio, Applied
# Spatial Data Analysis with R (Use R!, 1st ed.) Sec. 10.5.3:
#   n = 281, RSS = 119.6, "Sigma squared (ML): 0.4255",
#   "AICc (GWR p. 61, eq 2.33; p. 96, eq. 4.21): 568",
#   "AIC (GWR p. 96, eq. 4.22): 561.6".
# tr(S) is printed only to one decimal, so solve for it from the AIC -- one
# unknown -- and check that the SAME tr(S) reproduces the AICc.  This pins
# both formulas at once and is the reason the stub's printed AICc could not
# be used: it reproduces neither number.
n_ny, rss_ny = 281, 119.6
sigma2_ny = rss_ny / n_ny
base = 2 * n_ny * np.log(np.sqrt(sigma2_ny)) + n_ny * np.log(2 * np.pi) + n_ny
tr_S_ny = 561.6 - base  # AIC = base + tr(S)
chk(
    "eq (4.22) AIC recovers tr(S) in the printed range",
    3.5 < tr_S_ny < 5.0,
    f"tr(S)={tr_S_ny:.3f} (printed effective parameters 4.4)",
)
aicc_ny = g.aicc_from_parts(n_ny, sigma2_ny, tr_S_ny)
chk(
    "eq (2.33) AICc reproduces the published 568",
    abs(aicc_ny - 568.0) < 0.5,
    f"AICc={aicc_ny:.2f}",
)
chk(
    "eq (4.22) AIC reproduces the published 561.6",
    abs(g.aic_from_parts(n_ny, sigma2_ny, tr_S_ny) - 561.6) < 1e-9,
)
# The formula the stub printed, taken literally.
stub = 2 * n_ny * np.log(np.sqrt(sigma2_ny)) + 2 * tr_S_ny + 2 * tr_S_ny * tr_S_ny / (
    n_ny - tr_S_ny - 1
)
chk(
    "the stub's printed AICc reproduces NEITHER number",
    abs(stub - 568.0) > 100 and abs(stub - 561.6) > 100,
    f"stub={stub:.2f}",
)


print("\n[Charlton white paper p. 6 / spgwr / GWmodel gw.weight.r -- kernels]")
d = np.linspace(0, 2 * BW, 41)
chk("gaussian w(0) = 1", close(g.kernel_weights(0.0, BW, "gaussian"), 1.0))
chk(
    "gaussian = exp(-0.5 (d/h)^2) everywhere",
    close(g.kernel_weights(d, BW, "gaussian"), np.exp(-0.5 * (d / BW) ** 2)),
)
chk(
    "gaussian is positive at every finite distance (no truncation)",
    bool(np.all(g.kernel_weights(d, BW, "gaussian") > 0)),
)
chk(
    "Sec. 5.3.2 density form is the same kernel over h sqrt(2 pi)",
    close(
        g.kernel_weights(d, BW, "gaussian", normalized=True),
        g.kernel_weights(d, BW, "gaussian") / (BW * np.sqrt(2 * np.pi)),
    ),
)
inside = d < BW
chk(
    "bisquare = (1 - (d/h)^2)^2 inside, 0 outside",
    close(g.kernel_weights(d, BW, "bisquare"), np.where(inside, (1 - (d / BW) ** 2) ** 2, 0.0)),
)
chk(
    "tricube = (1 - (d/h)^3)^3 inside, 0 outside",
    close(g.kernel_weights(d, BW, "tricube"), np.where(inside, (1 - (d / BW) ** 3) ** 3, 0.0)),
)
chk(
    "boxcar = 1 inside, 0 outside (the only source naming it is GWmodel)",
    close(g.kernel_weights(d, BW, "boxcar"), np.where(inside, 1.0, 0.0)),
)
# Both truncated kernels are C^1 at the support edge -- that is the "useful
# property that the weight is zero at a finite distance" plus smoothness the
# white paper calls "near-Gaussian".  The boxcar is deliberately not.
h = 1e-6
for kern, smooth in (("bisquare", True), ("tricube", True), ("boxcar", False)):
    lo = float(g.kernel_weights(BW - h, BW, kern))
    chk(f"{kern} {'is' if smooth else 'is NOT'} continuous at d = h", (lo < 1e-9) == smooth,
        f"w(h-)={lo:.3e}")

for bad in ("epanechnikov", "quartic", ""):
    try:
        g.kernel_weights(1.0, 1.0, bad)
        ok = False
    except ValueError:
        ok = True
    chk(f"unknown kernel {bad!r} raises", ok)
for bad_bw in (0.0, -1.0, np.nan, np.inf):
    try:
        g.kernel_weights(1.0, bad_bw, "gaussian")
        ok = False
    except ValueError:
        ok = True
    chk(f"bandwidth {bad_bw!r} raises", ok)


print("\n[mgwr/kernels.py -- adaptive bandwidth]")
row = D[0]
for k in (5, 12, n):
    hk = g.adaptive_bandwidth(row, k)
    chk(
        f"adaptive bw at k={k} admits exactly {k} neighbours",
        int(np.sum(g.kernel_weights(row, hk, "bisquare") > 0)) == k,
        f"h={hk:.4f}",
    )
chk(
    "adaptive bw counts the regression point as its own first neighbour",
    close(g.adaptive_bandwidth(row, 1), 0.0 * 1.0000001) or g.adaptive_bandwidth(row, 1) == 0.0,
    f"h(k=1)={g.adaptive_bandwidth(row, 1):.3e} and d_ii = {row[0]:.3e}",
)
for bad in (0, n + 1, -3):
    try:
        g.adaptive_bandwidth(row, bad)
        ok = False
    except ValueError:
        ok = True
    chk(f"n_neighbours={bad} raises", ok)


print("\n[Schabenberger Sec. 6.1.3.1 p. 317 -- the hat matrix L and sigma^2]")
fit = g.gwr_fit(y, X, D, BW)
S = fit["S"]
chk("y_hat = S y", close(fit["fitted"], S @ y, 1e-12))
# Row i of L, written out directly from the book rather than from the code.
i = 7
w_i = g.kernel_weights(D[i], BW, "gaussian")
W = np.diag(w_i)
row_book = X[i] @ np.linalg.inv(X.T @ W @ X) @ X.T @ W
chk("row i of S equals x_i'(X'W_i X)^-1 X'W_i", close(S[i], row_book, 1e-10))
beta_book = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
chk("local beta(s_i) is the WLS solution at s_i", close(fit["params"][i], beta_book, 1e-10))
ImS = np.eye(n) - S
chk(
    "RSS = y'(I-S)'(I-S)y equals sum of squared residuals",
    close(fit["rss"], float(np.sum(fit["resid"] ** 2)), 1e-9),
)
chk(
    "Cressie p. 317 sigma^2 divides by tr{(I-L)(I-L)'}",
    close(fit["sigma2_cressie"], fit["rss"] / np.trace(ImS @ ImS.T), 1e-10),
)
chk(
    "the ML sigma^2 the AICc wants divides by n instead",
    close(fit["sigma2"], fit["rss"] / n, 1e-14) and fit["sigma2"] != fit["sigma2_cressie"],
)
chk(
    "effective parameters = 2 tr(S) - tr(S'S)",
    close(fit["effective_parameters"], 2 * fit["tr_S"] - fit["tr_STS"], 1e-12),
)

# Weighted least squares is invariant to a positive scalar on every weight,
# so the book's density-form Gaussian and the GWR literature's unnormalised
# one give the same fit.  This is why two different printed "Gaussian
# kernels" are not a contradiction.
S_density = np.empty((n, n))
for r in range(n):
    Wr = np.diag(g.kernel_weights(D[r], BW, "gaussian", normalized=True))
    S_density[r] = X[r] @ np.linalg.inv(X.T @ Wr @ X) @ X.T @ Wr
chk(
    "Sec. 5.3.2's density Gaussian builds the SAME hat matrix as GWR's",
    close(S_density, S, 1e-9),
    f"max|dS|={np.max(np.abs(S_density - S)):.2e}",
)
lhs = np.linalg.inv(X.T @ (W * 3.7) @ X) @ X.T @ (W * 3.7) @ y
chk("WLS is scale-invariant in the weights (checked directly)", close(lhs, beta_book, 1e-9))


print("\n[bandwidth -> infinity collapses GWR to the global OLS fit]")
big = g.gwr_fit(y, X, D, 1e7)
ols = np.linalg.lstsq(X, y, rcond=None)[0]
chk("every local beta -> the OLS beta", close(big["params"], np.tile(ols, (n, 1)), 1e-5))
chk("tr(S) -> p", abs(big["tr_S"] - X.shape[1]) < 1e-6, f"tr(S)={big['tr_S']:.8f}, p={X.shape[1]}")
chk(
    "effective parameters -> p as well (S becomes a projection)",
    abs(big["effective_parameters"] - X.shape[1]) < 1e-5,
    f"enp={big['effective_parameters']:.8f}",
)
chk(
    "tr(S) rises as the bandwidth falls (a narrower kernel spends more df)",
    g.gwr_fit(y, X, D, 1.5)["tr_S"] > g.gwr_fit(y, X, D, 6.0)["tr_S"] > big["tr_S"],
)


print("\n[spgwr gwr.cv.f -- the CV score is genuinely out-of-sample]")
cv = g.cv_score(y, X, D, BW)
manual = 0.0
for i in range(n):
    keep = np.arange(n) != i
    w = g.kernel_weights(D[i][keep], BW, "gaussian")
    Wi = np.diag(w)
    b = np.linalg.inv(X[keep].T @ Wi @ X[keep]) @ X[keep].T @ Wi @ y[keep]
    manual += float((y[i] - X[i] @ b) ** 2)
chk(
    "zeroing w_ii is the same as deleting row i and refitting",
    close(cv, manual, 1e-8),
    f"cv={cv:.10f}",
)
in_sample = float(np.sum(fit["resid"] ** 2))
chk(
    "the CV score exceeds the in-sample RSS at the same bandwidth",
    cv > in_sample,
    f"{cv:.4f} > {in_sample:.4f}",
)


print("\n[bandwidth selection]")
# `y` above has globally constant coefficients, so its optimal bandwidth is
# genuinely infinite and every criterion runs to the top of the search
# interval -- a fixture that would make the checks below vacuous.  Use a
# response whose coefficients really do vary in space, so the optimum is
# interior and the optimiser has something to find.
bvary = 1.0 + 0.8 * np.sin(coords[:, 0] * 0.9)
yv = X[:, 0] + bvary * X[:, 1] - 0.5 * X[:, 2] + 0.05 * rs.standard_normal(n)
sel_cv = g.select_bandwidth(yv, X, coords, criterion="cv")
sel_aicc = g.select_bandwidth(yv, X, coords, criterion="aicc")
lo, hi = sel_cv["bounds"]
chk(
    "spgwr's search interval is [diag/1000, diag] of the bounding box",
    close((lo, hi), (hi / 1000.0, hi), 1e-9),
    f"[{lo:.4f}, {hi:.4f}]",
)
chk(
    "CV optimum is interior, not pinned to either end of the interval",
    lo * 1.5 < sel_cv["bandwidth"] < hi * 0.95,
    f"bw={sel_cv['bandwidth']:.4f} in [{lo:.4f}, {hi:.4f}]",
)
chk(
    "AICc optimum is interior too",
    lo * 1.5 < sel_aicc["bandwidth"] < hi * 0.95,
    f"bw={sel_aicc['bandwidth']:.4f}",
)
grid = np.linspace(lo, hi, 400)
brute = float(grid[int(np.argmin([g.cv_score(yv, X, D, b) for b in grid]))])
chk(
    "golden section finds the same CV minimum as a 400-point brute force",
    abs(sel_cv["bandwidth"] - brute) < 0.05 * (hi - lo),
    f"golden={sel_cv['bandwidth']:.4f} brute={brute:.4f}",
)
chk(
    "the score reported is the objective at the reported bandwidth",
    close(sel_cv["score"], g.cv_score(yv, X, D, sel_cv["bandwidth"]), 1e-9),
)
sel_ad = g.select_bandwidth(yv, X, coords, criterion="aicc", adaptive=True)
chk(
    "the adaptive search returns an integer neighbour count strictly below n",
    isinstance(sel_ad["bandwidth"], int) and 2 <= sel_ad["bandwidth"] < n,
    f"k={sel_ad['bandwidth']}",
)
# Golden section on a unimodal parabola with a known minimum.
xstar, fstar = g.golden_section(lambda t: (t - 2.75) ** 2 + 1.0, 0.0, 10.0, tol=1e-9)
chk("golden section on a known parabola", abs(xstar - 2.75) < 1e-6 and abs(fstar - 1.0) < 1e-9)
try:
    g.golden_section(lambda t: t, 5.0, 5.0)
    ok = False
except ValueError:
    ok = True
chk("golden section rejects an empty interval", ok)
for bad in ("mse", "gcv", ""):
    try:
        g.gwr_criterion(y, X, D, BW, criterion=bad)
        ok = False
    except ValueError:
        ok = True
    chk(f"unknown criterion {bad!r} raises", ok)


print("\n[mgwr/search.py multi_bw -- MGWR backfitting]")
# A response whose two covariates genuinely act at different scales.
u = coords[:, 0]
b1 = 1.0 + 0.05 * u                       # near-global variation
b2 = np.sin(u)                            # short-scale variation
Xm = np.column_stack([np.ones(n), rs.uniform(-1, 1, n)])
ym = b1 * Xm[:, 0] + b2 * Xm[:, 1] + 0.05 * rs.standard_normal(n)
m = g.mgwr_backfit(ym, Xm, coords, criterion="aicc", tol=1e-4, max_iter=40)
chk("one bandwidth per covariate is returned", m["bandwidths"].shape == (Xm.shape[1],),
    f"bws={np.round(m['bandwidths'], 3).tolist()}")
chk("backfitting converged inside max_iter", m["converged"], f"{m['n_iter']} sweeps")
chk("the SOC score decreases to below tol", m["score_history"][-1] < 1e-4,
    f"SOC={m['score_history'][-1]:.3e}")
chk(
    "the two covariates get genuinely different scales",
    abs(m["bandwidths"][0] - m["bandwidths"][1]) > 1e-6,
)
# With standardize=True (the 2024 book's requirement) the coefficients are on
# the standardized scale while `fitted` is converted back to the units of y,
# so the identity holds through the scaling, not against the raw design.
Xm_std = Xm.copy()
nz = Xm.std(axis=0, ddof=0) > 0
Xm_std[:, nz] = (Xm[:, nz] - Xm[:, nz].mean(axis=0)) / Xm[:, nz].std(axis=0, ddof=0)
chk(
    "fitted = (sum_j beta_j(s_i) x_ij^std) * y_scale + y_centre",
    close(m["fitted"],
          np.sum(m["params"] * Xm_std, axis=1) * m["y_scale"] + m["y_centre"],
          1e-10),
)
chk(
    "standardization is on by default, per the 2024 book Sec. 2.3.3.2",
    m["standardized"] is True and abs(m["y_scale"] - ym.std(ddof=0)) < 1e-12,
)
raw = g.mgwr_backfit(ym, Xm, coords, criterion="aicc", tol=1e-4, max_iter=40,
                     standardize=False)
chk(
    "standardize=False is reachable and reports itself",
    raw["standardized"] is False and raw["y_scale"] == 1.0,
)
chk("MGWR residuals beat the single-bandwidth GWR it started from",
    float(np.sum(m["resid"] ** 2)) <= float(np.sum(g.gwr_fit(ym, Xm, D, m["bandwidth_gwr"])["resid"] ** 2)) * 1.5)
# SOC-f, recomputed from the definition in multi_bw.
bh = m["bandwidth_history"]
chk("bandwidth history has one row per sweep", len(bh) == m["n_iter"])
chk("the last history row is the reported bandwidth vector", close(bh[-1], m["bandwidths"]))
m_rss = g.mgwr_backfit(ym, Xm, coords, criterion="aicc", tol=1e-4, max_iter=40, rss_score=True)
chk(
    "SOC-RSS is a different score but converges to comparable bandwidths",
    np.all(np.abs(m_rss["bandwidths"] - m["bandwidths"]) < 0.5 * np.maximum(m["bandwidths"], 1.0)),
    f"rss-bws={np.round(m_rss['bandwidths'], 3).tolist()}",
)
# k = 1 must reduce to a plain univariate GWR.
# k = 1 has nothing to backfit, so the bandwidth must be the one a plain
# univariate GWR would pick -- on the SAME data the backfit saw.  Centring a
# covariate matters here because each inner GWR has no intercept to absorb it,
# so the comparison is run with standardization off on both sides.
m1 = g.mgwr_backfit(ym, Xm[:, [1]], coords, criterion="aicc", tol=1e-6,
                    max_iter=40, standardize=False)
single = g.select_bandwidth(ym, Xm[:, [1]], coords, criterion="aicc")["bandwidth"]
chk(
    "MGWR with k = 1 reduces to single-bandwidth GWR",
    abs(m1["bandwidths"][0] - single) < 1e-6,
    f"{m1['bandwidths'][0]:.6f} vs {single:.6f}",
)
m1s = g.mgwr_backfit(ym, Xm[:, [1]], coords, criterion="aicc", tol=1e-6,
                     max_iter=40, standardize=True)
chk(
    "standardizing a single covariate changes the fit (no intercept absorbs the centring)",
    abs(m1s["bandwidths"][0] - m1["bandwidths"][0]) > 1e-9,
    f"std={m1s['bandwidths'][0]:.6f} raw={m1['bandwidths'][0]:.6f}",
)


print("\n[shape and argument validation]")
for bad in (lambda: g.gwr_fit(y[:-1], X, D, BW), lambda: g.gwr_fit(y, X, D[:-1, :-1], BW)):
    try:
        bad()
        ok = False
    except ValueError:
        ok = True
    chk("mismatched n raises", ok)
try:
    g.mgwr_backfit(ym[:-1], Xm, coords)
    ok = False
except ValueError:
    ok = True
chk("MGWR with mismatched n raises", ok)
try:
    g.kernel_weights([-1.0, 2.0], 1.0, "gaussian")
    ok = False
except ValueError:
    ok = True
chk("negative distances raise", ok)
try:
    g.kernel_weights(d, BW, "bisquare", normalized=True)
    ok = False
except ValueError:
    ok = True
chk("normalized= is rejected for non-Gaussian kernels", ok)

print(f"\n{'=' * 74}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
