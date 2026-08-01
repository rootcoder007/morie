"""Verify _schab_glmm.py against identities stated in the two primary sources."""
import sys
import numpy as np

sys.path.insert(0, "/home/rootcoder/work/morie/src")
from morie.fn import _schab_glmm as gm
from morie.fn._rng import random_normal

FAIL = []


def check(name, cond, detail=""):
    print("  %-60s %s%s" % (name, "PASS" if cond else "FAIL",
                            "" if cond else "   <- " + str(detail)))
    if not cond:
        FAIL.append(name)


def raises(fn):
    try:
        fn()
    except (ValueError, np.linalg.LinAlgError):
        return True
    return False


print("\n[links] g, g^-1, g' and dmu/deta")
for kind, mu in (("log", 2.5), ("logit", 0.3), ("identity", 1.7)):
    eta = gm.link(mu, kind)
    check("%s: g^-1(g(mu)) = mu" % kind,
          np.isclose(gm.link(eta, kind, inverse=True), mu, rtol=1e-14))
    h = 1e-7
    num = (gm.link(mu + h, kind) - gm.link(mu - h, kind)) / (2 * h)
    check("%s: g'(mu) matches a numerical derivative" % kind,
          np.isclose(gm.link_derivative(mu, kind), num, rtol=1e-6),
          "%.10f vs %.10f" % (gm.link_derivative(mu, kind), num))
    check("%s: dmu/deta = 1/g'(mu)" % kind,
          np.isclose(gm.mu_eta(mu, kind), 1.0 / gm.link_derivative(mu, kind)))

print("\n[6.3.4] conditional specification")
rs = np.random.RandomState(11)
n = 40
X = np.column_stack([np.ones(n), rs.uniform(-1, 1, n)])
beta = np.array([0.4, 0.8])
s2S = 0.6
mm = gm.marginal_moments_lognormal(X, beta, s2S, sigma2=1.0)
naive = gm.naive_marginal_mean(X, beta, "log")
check("E[Z] = m(s) exp{sigma_S^2/2}", np.allclose(mm["mean"], mm["m"] * np.exp(s2S / 2)))
check("g^-1(x'beta) is NOT the marginal mean (Sec. 6.3.4)",
      not np.allclose(naive, mm["mean"]))
check("the gap is exactly the factor exp{sigma_S^2/2}",
      np.allclose(mm["mean"] / naive, np.exp(s2S / 2)),
      "ratio %.6f vs %.6f" % ((mm["mean"] / naive)[0], np.exp(s2S / 2)))
# the printed Var has m(s) where m(s)^2 belongs; the book's own Cov settles it
mmc = gm.marginal_moments_lognormal(X, beta, s2S, sigma2=1.0, rho=np.eye(n))
check("Var equals the diagonal of Cov plus E[Var(Z|S)]  (fixes the dropped square)",
      np.allclose(mm["variance"], np.diag(mmc["covariance"]) + mm["m"] * np.exp(s2S / 2)),
      "max dev %.3e" % np.max(np.abs(mm["variance"]
                                     - (np.diag(mmc["covariance"])
                                        + mm["m"] * np.exp(s2S / 2)))))
wrong = mm["m"] * np.exp(s2S) * (np.exp(s2S) - 1.0)      # the literal misprint
check("the literal (unsquared) printing disagrees, so the fix is load-bearing",
      not np.allclose(mm["variance"], mm["m"] * np.exp(s2S / 2) + wrong))
check("Var[Z] > E[Z] even at sigma^2 = 1 (Example 6.6 remark)",
      np.all(mm["variance"] > mm["mean"]))

print("\n[6.3.5] pseudo-likelihood")
d = np.abs(np.subtract.outer(np.linspace(0, 10, n), np.linspace(0, 10, n)))
Sigma_S = 0.5 * np.exp(-d / 3.0)
S_true = np.linalg.cholesky(Sigma_S + 1e-10 * np.eye(n)) @ random_normal(n, seed=7, stream=1)
mu_true = np.exp(X @ beta + S_true)
z = np.array([float(rs.poisson(m)) for m in mu_true])

nu = gm.pseudo_data(z, mu_true, "log")
check("eq (6.78) pseudo-data: log(mu) + (Z-mu)/mu for the log link",
      np.allclose(nu, np.log(mu_true) + (z - mu_true) / mu_true))
Sm = gm.sigma_mu(mu_true, 1.0, "poisson", "log")
check("eq (6.79) is diagonal when R = I", np.allclose(Sm, np.diag(np.diag(Sm))))
check("eq (6.79) diagonal is sigma^2 v(mu)/ (dmu/deta)^2 = 1/mu for Poisson-log",
      np.allclose(np.diag(Sm), 1.0 / mu_true), np.diag(Sm)[:3])

fit = gm.fit_pseudo_likelihood(z, X, Sigma_S, family="poisson")
check("PL converges", fit["converged"], fit["n_iter"])
check("PL recovers beta within 3 se", np.all(np.abs(fit["beta"] - beta) < 3 * fit["se_beta"]),
      "beta %s vs %s (se %s)" % (np.round(fit["beta"], 3), beta, np.round(fit["se_beta"], 3)))
sc = gm.pql_score(z, X, fit["beta"], fit["S"], Sigma_S, "poisson", "log")
check("PQL score for beta vanishes at the PL solution (Sec. 6.3.5.3 equivalence)",
      np.max(np.abs(sc["score_beta"])) < 1e-6, np.max(np.abs(sc["score_beta"])))
check("PQL score for S vanishes at the PL solution",
      np.max(np.abs(sc["score_S"])) < 1e-6, np.max(np.abs(sc["score_S"])))
r_obj = gm.reml_objective(X, fit["Sigma_nu"], fit["pseudo_data"])
check("eq (6.84) REML objective is finite", np.isfinite(r_obj), r_obj)
worse = gm.reml_objective(X, fit["Sigma_nu"] * 8.0, fit["pseudo_data"])
check("REML objective penalises a badly scaled Sigma_nu", worse > r_obj,
      "%.3f vs %.3f" % (worse, r_obj))

print("\n[6.3.6] prediction")
pr = gm.predict_glm(nu0_hat=1.2, sigma2_nu0=0.09, mu0_hat=3.0, link_kind="log")
check("eq (6.90) predictor", np.isclose(pr["prediction"], 3.0 + 3.0 * (1.2 - np.log(3.0))))
check("eq (6.91) MSPE = (dmu/deta)^2 sigma^2_nu", np.isclose(pr["mspe"], 9.0 * 0.09))
check("eq (6.87) inverse-link predictor is a DIFFERENT number",
      not np.isclose(pr["inverse_link_prediction"], pr["prediction"]),
      "%.4f vs %.4f" % (pr["inverse_link_prediction"], pr["prediction"]))
check("the result records which predictor the MSPE belongs to",
      "6.90" in pr["mspe_is_for"] and "6.87" in pr["mspe_is_for"])

print("\n[ICAR / LCAR / BYM]")
# 4x4 rook grid
A = np.zeros((4, 4))
for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
    A[i, j] = A[j, i] = 1.0
R = gm.neighbour_structure(A)
check("eq (4): diagonal is n_i", np.allclose(np.diag(R), A.sum(1)))
check("eq (4): off-diagonal is -1 for neighbours", R[0, 1] == -1 and R[0, 3] == 0)
check("R is singular: rows sum to zero (why eq 3 needs a pseudo-inverse)",
      np.allclose(R @ np.ones(4), 0.0))
check("R has rank deficiency 1", np.linalg.matrix_rank(R) == 3)
check("icar_covariance uses the Moore-Penrose inverse",
      np.allclose(gm.icar_covariance(R), np.linalg.pinv(R)))
u = np.array([0.5, -0.2, 0.1, -0.4])
fc = gm.icar_full_conditional(u, A)
check("eq (5): conditional mean is the neighbour average",
      np.isclose(fc["mean"][0], (u[1] + u[2]) / 2))
check("eq (5): conditional variance is sigma^2/n_i",
      np.allclose(fc["variance"], 1.0 / A.sum(1)))

Q0, _ = gm.lcar_precision(R, 0.0)
Q1, _ = gm.lcar_precision(R, 1.0)
check("eq (6): rho = 0 gives the exchangeable prior Q = I", np.allclose(Q0, np.eye(4)))
check("eq (6): rho = 1 gives the ICAR prior Q = R", np.allclose(Q1, R))
lfc1 = gm.lcar_full_conditional(u, A, 1.0)
check("eq (7) at rho = 1 collapses to the ICAR conditional (5)",
      np.allclose(lfc1["mean"], fc["mean"]) and np.allclose(lfc1["variance"], fc["variance"]))
lfc0 = gm.lcar_full_conditional(u, A, 0.0)
check("eq (7) at rho = 0 gives mean 0 and variance sigma^2",
      np.allclose(lfc0["mean"], 0.0) and np.allclose(lfc0["variance"], 1.0))
check("rho outside [0,1] rejected", raises(lambda: gm.lcar_precision(R, 1.5)))

v = np.array([0.05, 0.02, -0.03, 0.01])
check("BYM convolution is u + v", np.allclose(gm.bym_convolution(u, v), u + v))
check("BYM identifiability is reported, not assumed away",
      "not separately identifiable" in gm.bym_identifiability_note())
check("asymmetric adjacency rejected",
      raises(lambda: gm.neighbour_structure(np.array([[0, 1], [0, 0]]))))
check("self-neighbours rejected",
      raises(lambda: gm.neighbour_structure(np.array([[1.0, 1], [1, 1]]))))

print("\n[disease mapping]")
E = np.array([10.0, 20.0, 5.0, 8.0])
Z = np.array([12.0, 18.0, 7.0, 8.0])
check("SMR = Z/E (the MLE of the relative risk, Sec. 6.4.3.2)",
      np.allclose(gm.smr(Z, E), Z / E))
mu_d = gm.poisson_disease_mean(E, np.ones((4, 1)), np.array([0.1]), u)
check("eq (6.101): E exp{x'beta + psi}, with log E as an offset",
      np.allclose(mu_d, E * np.exp(0.1 + u)))

print("\n[temporal + space-time]")
R1 = gm.random_walk_structure(6, 1)
R2 = gm.random_walk_structure(6, 2)
check("RW1 structure has rank deficiency 1 (the constant)",
      6 - np.linalg.matrix_rank(R1) == 1)
check("RW2 structure has rank deficiency 2 (constant and slope)",
      6 - np.linalg.matrix_rank(R2) == 2)
check("RW1 annihilates a constant", np.allclose(R1 @ np.ones(6), 0))
check("RW2 annihilates a linear trend", np.allclose(R2 @ np.arange(6.0), 0))

# Table 1 ranks, with I = 4 areas and T = 6 periods
for kind, expected in (("I", 4 * 6), ("II", 4 * (6 - 1)),
                       ("III", (4 - 1) * 6), ("IV", (4 - 1) * (6 - 1))):
    inter = gm.interaction_structure(R, R1, kind)
    check("Table 1 Type %-3s rank = %d (RW1)" % (kind, expected),
          inter["rank"] == expected, "got %d" % inter["rank"])
    con = gm.null_space_constraints(inter["structure"])
    check("Type %-3s constraints = rank deficiency" % kind,
          con["n_constraints"] == inter["rank_deficiency"],
          "%d vs %d" % (con["n_constraints"], inter["rank_deficiency"]))
i1 = gm.interaction_structure(R, R1, "I")
check("only Type I needs no constraints (it is full rank)",
      i1["rank_deficiency"] == 0)
i4 = gm.interaction_structure(R, R1, "IV")
c4 = gm.null_space_constraints(i4["structure"])
dd = gm.apply_sum_to_zero(np.arange(24.0), c4["A"])
check("eq (12): the constrained delta satisfies A delta = 0",
      np.max(np.abs(c4["A"] @ dd)) < 1e-9, np.max(np.abs(c4["A"] @ dd)))
r2i = gm.interaction_structure(R, gm.random_walk_structure(6, 2), "II")
check("Table 1 Type II with RW2 has rank I(T-2)", r2i["rank"] == 4 * (6 - 2),
      r2i["rank"])

lt = gm.linear_trend_log_risk(0.1, u, 0.05, np.array([0.01, 0, -0.01, 0.02]),
                              np.arange(6.0))
check("eq (9) shape is (areas, periods)", lt.shape == (4, 6), lt.shape)
check("eq (9) at t=0 is alpha + u_i", np.allclose(lt[:, 0], 0.1 + u))
npr = gm.nonparametric_log_risk(0.1, u, np.zeros(6), np.arange(6.0))
check("eq (10) additive form (no interaction)",
      np.allclose(npr, 0.1 + u[:, None] + np.arange(6.0)[None, :]))
check("eq (10) rejects a misshaped interaction term",
      raises(lambda: gm.nonparametric_log_risk(0.1, u, np.zeros(6),
                                               np.arange(6.0), np.zeros((3, 3)))))

print("\n[BYM 1991] the convolution model, Besag/York/Mollie Sec. 4")
rsb = np.random.RandomState(5)
# a 12-area chain-with-branches adjacency
nb = 12
Ab = np.zeros((nb, nb))
for i in range(nb - 1):
    Ab[i, i + 1] = Ab[i + 1, i] = 1.0
Ab[0, 5] = Ab[5, 0] = 1.0
Ab[3, 9] = Ab[9, 3] = 1.0
Eb = rsb.uniform(5, 40, nb)
u_t = np.linspace(-0.4, 0.4, nb)
yb = np.array([float(rsb.poisson(e * np.exp(x))) for e, x in zip(Eb, u_t)])

kap, laam = 0.129, 0.011                 # the paper's thyroid-cancer estimates
m = gm.bym_map(yb, Eb, Ab, kap, laam)
check("BYM MAP converges (log posterior is strictly concave, Sec. 4)",
      m["converged"], m["n_iter"])
check("sum v* = 0  (stated in Sec. 4, here a consequence of stationarity)",
      abs(m["sum_v"]) < 1e-8, m["sum_v"])
check("sum c_i exp(u*_i+v*_i) = sum y_i  (fitted total matches observed)",
      abs(m["fitted_total"] - m["observed_total"]) < 1e-7,
      "%.9f vs %.9f" % (m["fitted_total"], m["observed_total"]))
check("relative risk is exp(u* + v*)", np.allclose(m["relative_risk"], np.exp(m["x"])))
# concavity => the optimum beats perturbations in every direction tried
lp0 = m["log_posterior"]
worse = []
for _ in range(12):
    du = rsb.normal(0, 0.05, nb)
    dv = rsb.normal(0, 0.02, nb)
    worse.append(gm.bym_log_posterior(yb, Eb, m["u"] + du, m["v"] + dv,
                                      kap, laam, Ab))
check("no perturbation beats the MAP (single maximum)", max(worse) < lp0,
      "best perturbation %.6f vs MAP %.6f" % (max(worse), lp0))
# starting elsewhere must reach the same point -- strict concavity
m2 = gm.bym_map(yb * 1.0, Eb, Ab, kap, laam, max_iter=400)
check("the maximum is unique (same solution from the same convex problem)",
      np.allclose(m["x"], m2["x"], atol=1e-9))

check("eq (4.2) ICAR log prior is invariant to adding a constant (improper)",
      np.isclose(gm.bym_icar_log_prior(u_t, Ab, kap),
                 gm.bym_icar_log_prior(u_t + 3.7, Ab, kap)),
      "the density addresses only differences, not the overall level")
check("eq (4.3) conditional moments match icar_full_conditional at sigma^2=kappa",
      np.allclose(gm.bym_icar_conditional_moments(u_t, Ab, kap)["variance"],
                  gm.icar_full_conditional(u_t, Ab, kap)["variance"]))
check("eq (4.4) median prior is also translation invariant",
      np.isclose(gm.bym_median_log_prior(u_t, Ab, kap),
                 gm.bym_median_log_prior(u_t + 2.0, Ab, kap)))
check("eq (4.4) penalises absolute differences, (4.2) squared ones",
      not np.isclose(gm.bym_median_log_prior(u_t, Ab, 1.0),
                     gm.bym_icar_log_prior(u_t, Ab, 1.0)))
# kappa -> 0 forces u constant (Sec. 4 reading of the scale parameters)
m_small = gm.bym_map(yb, Eb, Ab, 1e-6, laam)
check("kappa -> 0 drives u* to a constant (Sec. 4)",
      float(np.ptp(m_small["u"])) < 1e-3, np.ptp(m_small["u"]))
m_lam0 = gm.bym_map(yb, Eb, Ab, kap, 1e-8)
check("lambda -> 0 drives v* to zero (Sec. 4)",
      float(np.max(np.abs(m_lam0["v"]))) < 1e-5, np.max(np.abs(m_lam0["v"])))
check("eq (4.5) rejects non-positive expected counts",
      raises(lambda: gm.bym_log_posterior(yb, -Eb, u_t, u_t, kap, laam, Ab)))
check("eq (4.6) epsilon term is present in the posterior",
      not np.isclose(gm.bym_log_posterior(yb, Eb, m["u"], m["v"], kap, laam, Ab, epsilon=0.01),
                     gm.bym_log_posterior(yb, Eb, m["u"], m["v"], kap, laam, Ab, epsilon=0.0)))

print("\n%s  %d failed" % ("=" * 66, len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
sys.exit(1 if FAIL else 0)
