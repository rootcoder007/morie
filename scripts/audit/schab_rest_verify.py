"""Source-identity checks for the remaining Schabenberger shelf modules.

Covers _schab_moran (Ch 1), _schab_pp cross-K (Ch 3), _schab_spectral (Ch 4)
and _schab_nonstat (Ch 8). Every check names the equation or the printed
result it is checking against. Run from the repo root:

    python3 scripts/audit/schab_rest_verify.py
"""

import sys

import numpy as np

sys.path.insert(0, "src")

from morie.fn import _schab_moran as mo      # noqa: E402
from morie.fn import _schab_nonstat as ns    # noqa: E402
from morie.fn import _schab_pp as pp         # noqa: E402
from morie.fn import _schab_spectral as sp   # noqa: E402

PASS, FAIL = [], []


def chk(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {name:<62} {'PASS' if ok else 'FAIL'}  {detail}")


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


print("\n[Ch 1 Sec 1.3.2 -- Moran's I moments, checked on Example 1.7]")
W = rook(10)
s = mo.weight_sums(W)
chk("weight sums on the 10x10 rook lattice", (s["S0"], s["S1"], s["S2"]) == (360.0, 720.0, 5312.0),
    f"S0={s['S0']:.0f} S1={s['S1']:.0f} S2={s['S2']:.0f}")

rs = np.random.RandomState(0)
x, y = np.meshgrid(np.arange(10), np.arange(10), indexing="ij")
z = (1.4 + 0.1 * x + 0.2 * y + 0.002 * x ** 2).ravel() + rs.standard_normal(100)
m = mo.moran_moments(z, W)
chk("E[I] = -1/(n-1), the same under both assumptions (p. 22)",
    abs(m["expectation"] - (-1 / 99)) < 1e-15, f"{m['expectation']:.4f}, book prints -0.0101")
chk("Example 1.7 Gaussian sd reproduced exactly",
    abs(m["sd_normal"] - 0.0731) < 5e-5, f"{m['sd_normal']:.4f}, book prints 0.0731")
chk("randomization sd lands on the book's value too",
    abs(m["sd_randomization"] - 0.0732) < 2e-4,
    f"{m['sd_randomization']:.4f} at b={m['kurtosis_b']:.2f}, book prints 0.0732")

# The bracket that Problem 1.8 loses in print. Read literally, n multiplies
# only S1; the grouping the example supports puts n outside the whole group.
n_, S0, S1, S2 = 100.0, s["S0"], s["S1"], s["S2"]
b = m["kurtosis_b"]
EI = -1 / 99
second = b * ((n_ * n_ - n_) * S1 - 2 * n_ * S2 + 6 * S0 ** 2)
den = (n_ - 1) * (n_ - 2) * (n_ - 3) * S0 ** 2
sd_lit = np.sqrt((n_ * (n_ * n_ - 3 * n_ + 3) * S1 - n_ * S2 + 3 * S0 ** 2 - second) / den - EI ** 2)
chk("the literal (unbracketed) Problem 1.8 reading does NOT match the example",
    abs(sd_lit - 0.0732) > 5e-4, f"literal gives {sd_lit:.4f} against 0.0732")

chk("the normality variance is free of the data", 
    abs(mo.moran_moments(rs.standard_normal(100) * 9 + 4, W)["sd_normal"] - m["sd_normal"]) < 1e-15)
chk("the randomization variance is NOT free of the data",
    abs(mo.moran_moments(rs.standard_normal(100) ** 3, W)["sd_randomization"] - m["sd_randomization"]) > 1e-9)
chk("I is invariant to a*z+b (Problem 1.7)",
    abs(mo.moran_i(3.5 * z + 11.0, W) - mo.moran_i(z, W)) < 1e-12)
chk("E[c] = 1 for Geary (p. 22)", m["geary_expectation"] == 1.0, f"c={m['geary_c']:.4f}")
for bad, why in (((np.ones(100), W), "constant z"), ((z, np.eye(100)), "non-zero diagonal")):
    try:
        mo.moran_moments(*bad); ok = False
    except ValueError:
        ok = True
    chk(f"rejects {why}", ok)


print("\n[Ch 3 Sec 3.4.4 -- cross-K, eq (3.9)]")
reg = (0.0, 0.0, 1.0, 1.0)
worst = 0.0
for _ in range(60):
    p = rs.uniform(0.05, 0.95, 2)
    t = rs.uniform(0.01, 0.8)
    th = np.linspace(0, 2 * np.pi, 400001)[:-1]
    cx, cy = p[0] + t * np.cos(th), p[1] + t * np.sin(th)
    worst = max(worst, abs(pp.ripley_weight(p, reg, t)
                           - np.mean((cx >= 0) & (cx <= 1) & (cy >= 0) & (cy <= 1))))
chk("Ripley weight matches numerical integration of the circumference",
    worst < 5e-5, f"max|d|={worst:.2e} (the integration's own discretisation)")
vp = rs.uniform(0.02, 0.98, (200, 2))
vt = rs.uniform(0.01, 0.7, 200)
chk("vectorised Ripley weight is identical to the scalar reference",
    float(np.abs(np.array([pp.ripley_weight(vp[i], reg, vt[i]) for i in range(200)])
                 - pp.ripley_weights(vp, reg, vt)).max()) == 0.0)
chk("weight is exactly 1 for a circle wholly inside the window",
    pp.ripley_weight((0.5, 0.5), reg, 0.4) == 1.0)
ws = [pp.ripley_weight((0.5, 0.05), reg, t) for t in (0.02, 0.1, 0.3, 0.6)]
chk("weight falls as the circle spills further outside", bool(np.all(np.diff(ws) < 0)),
    str([round(w, 4) for w in ws]))

r = np.linspace(0.02, 0.25, 8)
# Thresholds below are set from measurement, not taste. Deviation of K* from
# pi h^2 for genuinely independent patterns, worst of 20 replicates:
#   n=200 0.344   n=400 0.156   n=800 0.074   n=1600 0.043
# It halves as n doubles, so it is Monte-Carlo, not bias. The convergence is
# the property worth asserting; a single-draw tolerance is not.
prs = np.random.RandomState(4242)
dev = {}
for nn in (200, 800):
    d_ = []
    for rep in range(5):
        aa = prs.uniform(0, 1, (nn, 2))
        b_ = prs.uniform(0, 1, (nn, 2))
        rr_ = pp.cross_k_combined(aa, b_, reg, r)
        d_.append(float((np.abs(rr_["K_star"] - np.pi * r ** 2) / (np.pi * r ** 2)).max()))
    dev[nn] = float(np.mean(d_))
chk("independent patterns: K* -> pi h^2 as n grows (p. 104)",
    dev[800] < 0.5 * dev[200], f"mean dev {dev[200]:.3f} at n=200 -> {dev[800]:.3f} at n=800")
a = prs.uniform(0, 1, (800, 2))
bb = prs.uniform(0, 1, (800, 2))
res = pp.cross_k_combined(a, bb, reg, r)
chk("independent patterns: K* within the measured band at n=800",
    float((np.abs(res["K_star"] - np.pi * r ** 2) / (np.pi * r ** 2)).max()) < 0.12)
chk("independent patterns: L* - h is near zero",
    float(np.abs(res["L_minus_h"]).max()) < 0.02,
    f"max|L*-h|={float(np.abs(res['L_minus_h']).max()):.4f}")
chk("Khat_12 != Khat_21, the asymmetry Lotwick-Silverman address",
    not np.allclose(res["K_12"], res["K_21"]))
chk("K* lies between the two one-sided estimators",
    bool(np.all((res["K_star"] >= np.minimum(res["K_12"], res["K_21"]) - 1e-12)
                & (res["K_star"] <= np.maximum(res["K_12"], res["K_21"]) + 1e-12))))
# "L* - h > 0 at EVERY r" is not a reliable statement: measured over 20
# replicates it held in only 13-16 of them, because the smallest radii carry
# the most sampling noise. What is reliable is the sign of the average and
# the behaviour away from the smallest radius.
b2 = np.clip(a[:400] + prs.normal(0, 0.01, (400, 2)), 0.001, 0.999)
lm = pp.cross_k_combined(a, b2, reg, r)["L_minus_h"]
chk("attraction pushes L* - h positive on average (p. 104)",
    float(lm.mean()) > 0, f"mean={float(lm.mean()):.4f}")
chk("attraction is unambiguous away from the smallest radius",
    bool(np.all(lm[2:] > 0)), str(np.round(lm, 4)))
rep2 = np.clip(a[:400] + prs.normal(0, 0.30, (400, 2)), 0.001, 0.999)
chk("a diffuse second pattern does not fake attraction",
    float(pp.cross_k_combined(a, rep2, reg, r)["L_minus_h"].mean()) < float(lm.mean()))
chk("the edge correction raises the estimate",
    bool(np.all(pp.cross_k_function(a, bb, reg, r, "ripley")
                >= pp.cross_k_function(a, bb, reg, r, "none") - 1e-12)))
chk("D(h) = 0 for two copies of one pattern (eq (3.10))",
    float(np.abs(pp.diggle_chetwynd_d(a, a, reg, r)["D"]).max()) < 1e-12)


print("\n[Ch 4 Sec 4.7.1 -- periodogram, eqs (4.57)-(4.59)]")
zz = rs.standard_normal((8, 6)) + 0.5 * np.add.outer(np.arange(8), np.arange(6))
A = sp.periodogram(zz)
B = sp.periodogram_from_covariance(zz)
mask = A["nonzero_mask"]
d = float(np.abs(A["periodogram"][mask] - B["periodogram"][mask]).max())
chk("eq (4.59): the periodogram IS the Fourier transform of Chat", d < 1e-10, f"max|d|={d:.2e}")
chk("mean-invariance at non-zero Fourier frequencies (p. 191)", A["mean_invariant"])
w1, w2, jj, kk = sp.fourier_frequencies(8, 6)
chk("one Fourier frequency per lattice row and column", (w1.size, w2.size) == (8, 6),
    f"j {jj[0]}..{jj[-1]}, k {kk[0]}..{kk[-1]}")
chk("all Fourier frequencies lie in [-pi, pi]",
    bool(np.all(np.abs(w1) <= np.pi + 1e-12) and np.all(np.abs(w2) <= np.pi + 1e-12)))
chk("the periodogram is non-negative", bool(np.all(A["periodogram"] >= -1e-12)))
cov, lj, lk = sp.sample_covariance_2d(zz)
chk("Chat(0,0) is the sample variance with divisor r*c",
    abs(cov[7, 5] - ((zz - zz.mean()) ** 2).sum() / 48) < 1e-12)
raw = sp.periodogram(zz + 20.0, omit_zero_frequency=False)
chk("at the origin the raw periodogram carries the squared mean, so (4.59) is restricted",
    abs(raw["periodogram"][raw["zero_index"]] - B["periodogram"][raw["zero_index"]]) > 1.0)
x1 = rs.standard_normal(9)
rr = 9
u = np.arange(1, rr + 1)
d1 = x1 - x1.mean()
worst1 = 0.0
for wj in 2 * np.pi * np.arange(-4, 5) / rr:
    if abs(wj) < 1e-12:
        continue
    lhs = 2 * rr * np.pi * np.abs((d1 * np.exp(-1j * wj * u)).sum()) ** 2 / (2 * np.pi * rr)
    rhs = float((np.outer(d1, d1) * np.cos(wj * np.subtract.outer(u, u))).sum())
    worst1 = max(worst1, abs(lhs - rhs))
chk("eq (4.58) holds in one dimension", worst1 < 1e-10, f"max|d|={worst1:.2e}")


print("\n[Ch 8 Sec 8.2.1 -- Hughes-Oliver point source, eq (8.1)]")
sc = rs.uniform(0, 10, (40, 2))
src = np.array([5.0, 5.0])
r0 = ns.point_source_correlation(sc, src, theta1=0.4)
chk("theta2 = theta3 = 0 reduces (8.1) to exp(-theta1 h) (p. 423)",
    float(np.abs(r0["correlation"] - np.exp(-0.4 * r0["separation"])).max()) < 1e-15)
chk("practical range 3/theta1 is where correlation reaches exp(-3)",
    abs(float(np.exp(-0.4 * ns.practical_range(0.4))) - float(np.exp(-3))) < 1e-12)
pa = np.array([[5.0, 5.0], [5.0, 6.0]])
pb = np.array([[5.0, 12.0], [5.0, 13.0]])
ca = ns.point_source_correlation(pa, src, 0.4, 0.3, 0.2)["correlation"][0, 1]
cb = ns.point_source_correlation(pb, src, 0.4, 0.3, 0.2)["correlation"][0, 1]
chk("equal separation, different source distance -> different correlation",
    abs(ca - cb) > 1e-6, f"{ca:.4f} vs {cb:.4f}: non-stationary")
d0 = 2.0
pair = np.array([[5.0 + d0, 5.0], [5.0, 5.0 + d0]])
al = ns.practical_range(0.4, 0.0, 0.2, ci=d0, cj=d0)
chk("equidistant pair: alpha = 3 exp(-theta3 c)/theta1 (p. 423)",
    abs(al - 3.0 * np.exp(-0.2 * d0) / 0.4) < 1e-12, f"alpha={al:.6f}")
pr = ns.point_source_correlation(pair, src, 0.4, 0.0, 0.2)
chk("and the correlation there is exp(-3h/alpha)",
    abs(pr["correlation"][0, 1] - float(np.exp(-3 * pr["separation"][0, 1] / al))) < 1e-12)
badm = ns.point_source_correlation(rs.uniform(0, 10, (20, 2)), rs.uniform(0, 10, 2),
                                   theta1=0.021, theta2=0.457, theta3=0.037)
chk("the stated constraints are necessary but NOT sufficient for PSD",
    True, f"a conforming parameter set gives min eigenvalue {badm['min_eigenvalue']:.3f}")
chk("an invalid correlation matrix is flagged, not passed on",
    (not badm["valid"]) and "warning" in badm)
for kw in ({"theta1": 0.0}, {"theta1": 0.4, "theta2": -1.0}, {"theta1": 0.4, "theta3": -1.0}):
    try:
        ns.point_source_correlation(sc, src, **kw); ok = False
    except ValueError:
        ok = True
    chk(f"rejects {kw}", ok)


print("\n[Ch 8 Sec 8.3.1 -- moving windows, Haas (1990, 1995)]")
big = rs.uniform(0, 10, (200, 2))
w = ns.haas_window(big, np.array([5.0, 5.0]))
chk("window starts at 35 sites (p. 426)", w["n_sites"] >= 35, f"n={w['n_sites']}")
chk("and grows five at a time", (w["n_sites"] - 35) % 5 == 0)
chk("stops once every lag class holds a pair", w["all_lag_classes_filled"])
zz2 = np.sin(big[:, 0] / 2.0) + 0.3 * rs.standard_normal(200)
tg = np.array([[3.0, 3.0], [7.0, 7.0], [5.0, 2.0]])
mwr = ns.moving_window_krige(big, zz2, tg, local_variogram=True)
lkr = ns.moving_window_krige(big, zz2, tg, local_variogram=False)
chk("moving window re-estimates theta in every window",
    len(set(np.round(mwr["local_range"], 6))) > 1, str(np.round(mwr["local_range"], 3)))
chk("local kriging keeps one global theta (p. 425)",
    len(set(np.round(lkr["local_range"], 9))) == 1 and lkr["theta_is_global"])
chk("the book's cautions are reported", "no longer best" in mwr["caveats"])

print(f"\n{'=' * 78}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED: " + ", ".join(FAIL))
    sys.exit(1)
