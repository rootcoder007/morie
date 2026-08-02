"""Copula cluster: _copula core, copgau copt copcla copgmb copfra
copjoe plkt taukcp spcoef blncop ginicop copExt clyfr copfr.

Anchored on Czado (2019) Table 3.2 p.54 (read in the PDF) plus the
copula axioms every family must satisfy."""

from morie.fn import _array_core as np
import pytest

from morie.fn._copula import FAMILIES, copula_cdf, copula_tau, tau_to_theta
from morie.fn.blncop import blomqvists_beta_copula
from morie.fn.clyfr import clayton_copula_frailty
from morie.fn.copcla import clayton_copula
from morie.fn.copExt import extremal_copula
from morie.fn.copfr import copula_frailty
from morie.fn.copfra import frank_copula
from morie.fn.copgau import gaussian_copula
from morie.fn.copgmb import gumbel_copula
from morie.fn.copjoe import joe_copula
from morie.fn.copt import t_copula
from morie.fn.ginicop import ginis_gamma_copula
from morie.fn.plkt import plackett_copula
from morie.fn.spcoef import spearmans_rho_copula
from morie.fn.taukcp import kendalls_tau_copula

# family -> (theta, nu); nu is only meaningful for the t copula
PARAMS = {
    "independence": (None, None),
    "gaussian": (0.6, None),
    "t": (0.6, 6.0),
    "clayton": (2.0, None),
    "gumbel": (2.0, None),
    "frank": (5.0, None),
    "joe": (2.5, None),
    "plackett": (4.0, None),
}


def test_every_family_satisfies_the_copula_axioms():
    u = np.array([0.1, 0.35, 0.5, 0.77, 0.9])
    for fam, (th, nu) in PARAMS.items():
        # grounded margins and uniform margins
        assert copula_cdf(fam, u, np.ones_like(u), th, nu) == pytest.approx(u, abs=1e-5)
        assert copula_cdf(fam, np.ones_like(u), u, th, nu) == pytest.approx(u, abs=1e-5)
        assert np.allclose(copula_cdf(fam, u, np.zeros_like(u), th, nu), 0.0, atol=1e-5)
        # Frechet-Hoeffding bounds
        U, V = np.meshgrid(u, u, indexing="ij")
        C = copula_cdf(fam, U, V, th, nu)
        assert np.all(C <= np.minimum(U, V) + 1e-6)
        assert np.all(C >= np.maximum(U + V - 1, 0.0) - 1e-6)
        # 2-increasing (nonnegative rectangle mass)
        rect = C[1:, 1:] - C[1:, :-1] - C[:-1, 1:] + C[:-1, :-1]
        assert np.all(rect >= -1e-8), fam


def test_tau_table_3_2_closed_forms():
    # Czado Table 3.2, p. 54 -- read in the PDF
    assert copula_tau("gaussian", 0.5) == pytest.approx(2 / np.pi * np.arcsin(0.5))
    assert copula_tau("t", 0.5) == pytest.approx(copula_tau("gaussian", 0.5))
    assert copula_tau("gumbel", 4.0) == pytest.approx(1 - 1 / 4)
    assert copula_tau("clayton", 2.0) == pytest.approx(2 / (2 + 2))
    assert copula_tau("independence", None) == 0.0
    # limiting cases: all families collapse to tau = 0 at independence
    assert copula_tau("gumbel", 1.0) == pytest.approx(0.0)
    assert copula_tau("joe", 1.0) == pytest.approx(0.0)
    assert copula_tau("clayton", 1e-8) == pytest.approx(0.0, abs=1e-6)
    assert copula_tau("frank", 1e-6) == pytest.approx(0.0, abs=1e-5)
    # Frank is the only Archimedean here admitting negative dependence
    assert copula_tau("frank", -5.0) == pytest.approx(-copula_tau("frank", 5.0))


def test_tau_matches_the_numeric_double_integral():
    # tau = 4 int int C dC - 1 must agree with the closed forms
    from morie.fn._copula import _tau_numeric

    for fam, th in (("clayton", 2.0), ("gumbel", 2.0), ("frank", 5.0), ("joe", 2.5)):
        assert _tau_numeric(fam, th, n=160) == pytest.approx(copula_tau(fam, th), abs=0.02)


def test_tau_to_theta_roundtrip():
    for fam in ("gaussian", "clayton", "gumbel", "frank", "joe", "plackett"):
        for tau in (0.15, 0.45, 0.7):
            th = tau_to_theta(fam, tau)
            assert copula_tau(fam, th) == pytest.approx(tau, abs=1e-6)
    assert tau_to_theta("frank", -0.4) < 0
    with pytest.raises(ValueError):
        tau_to_theta("clayton", -0.3)  # Clayton cannot be negative
    with pytest.raises(ValueError):
        tau_to_theta("gaussian", 1.5)


def test_family_front_ends_agree_with_the_core():
    u, v = 0.3, 0.7
    assert gaussian_copula(u, v, 0.5)["cdf"] == pytest.approx(copula_cdf("gaussian", u, v, 0.5))
    assert clayton_copula(u, v, 2.0)["cdf"] == pytest.approx(copula_cdf("clayton", u, v, 2.0))
    assert gumbel_copula(u, v, 2.0)["cdf"] == pytest.approx(copula_cdf("gumbel", u, v, 2.0))
    assert frank_copula(u, v, 5.0)["cdf"] == pytest.approx(copula_cdf("frank", u, v, 5.0))
    assert joe_copula(u, v, 2.5)["cdf"] == pytest.approx(copula_cdf("joe", u, v, 2.5))
    assert plackett_copula(u, v, 4.0)["cdf"] == pytest.approx(copula_cdf("plackett", u, v, 4.0))
    assert clayton_copula(u, v, 2.0)["tau"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        clayton_copula(u, v, -1.0)
    with pytest.raises(ValueError):
        gumbel_copula(u, v, 0.5)


def test_plackett_reduces_to_independence_at_one():
    u = np.array([0.2, 0.5, 0.8])
    U, V = np.meshgrid(u, u, indexing="ij")
    assert copula_cdf("plackett", U, V, 1.0) == pytest.approx(U * V)
    assert plackett_copula(0.4, 0.6, 1.0)["tau"] == pytest.approx(0.0, abs=0.02)


def test_copt_approaches_gaussian_as_nu_grows():
    u, v, rho = 0.4, 0.65, 0.5
    g = copula_cdf("gaussian", u, v, rho)
    near = t_copula(u, v, rho, nu=200.0)["cdf"]
    far = t_copula(u, v, rho, nu=2.0)["cdf"]
    assert abs(near - g) < abs(far - g)
    assert t_copula(u, v, rho, nu=5.0)["tau"] == pytest.approx(2 / np.pi * np.arcsin(rho))
    with pytest.raises(ValueError):
        t_copula(u, v, rho, nu=0.0)


def test_dependence_measures_order_correctly():
    # all four measures rise with the dependence parameter
    taus, rhos, betas, gammas = [], [], [], []
    for th in (1.2, 2.0, 5.0):
        taus.append(kendalls_tau_copula("gumbel", th)["tau"])
        rhos.append(spearmans_rho_copula("gumbel", th)["rho_s"])
        betas.append(blomqvists_beta_copula("gumbel", th)["beta"])
        gammas.append(ginis_gamma_copula("gumbel", th)["gamma"])
    for seq in (taus, rhos, betas, gammas):
        assert seq[0] < seq[1] < seq[2]
    # independence sends every measure to zero
    assert kendalls_tau_copula("independence")["tau"] == 0.0
    assert spearmans_rho_copula("independence")["rho_s"] == 0.0
    assert blomqvists_beta_copula("independence")["beta"] == pytest.approx(0.0)
    assert ginis_gamma_copula("independence")["gamma"] == pytest.approx(0.0, abs=1e-6)
    # rho_s > tau for these copulas, and the elliptical rho is exact
    assert spearmans_rho_copula("gaussian", 0.6)["exact"] is True
    assert spearmans_rho_copula("gaussian", 0.6)["rho_s"] == pytest.approx(
        6 / np.pi * np.arcsin(0.3)
    )
    assert spearmans_rho_copula("clayton", 2.0)["exact"] is False


def test_taukcp_roundtrip_field():
    out = kendalls_tau_copula("clayton", 3.0)
    assert out["tau"] == pytest.approx(0.6)
    assert out["theta_roundtrip"] == pytest.approx(3.0, abs=1e-6)
    with pytest.raises(ValueError):
        kendalls_tau_copula("weibull", 2.0)


def test_copExt_is_max_stable():
    u, v = 0.4, 0.7
    for A, th in (("gumbel", 2.0), ("galambos", 1.5)):
        c = extremal_copula(u, v, A, th)
        assert c["valid_pickands"] is True
        for k in (0.5, 2.0, 3.0):
            ck = extremal_copula(u**k, v**k, A, th)["cdf"]
            assert ck == pytest.approx(c["cdf"] ** k, rel=1e-8)  # max-stability
    # the logistic Pickands with theta = 1 is independence
    ind = extremal_copula(u, v, "gumbel", 1.0)
    assert ind["cdf"] == pytest.approx(u * v)
    # Gumbel via Pickands equals the Archimedean Gumbel (Czado Table 3.1)
    assert extremal_copula(u, v, "gumbel", 2.0)["cdf"] == pytest.approx(
        copula_cdf("gumbel", u, v, 2.0)
    )
    with pytest.raises(ValueError):
        extremal_copula(u, v, "weibull", 2.0)


def _paired_survival(seed=0, n=200, theta=2.0):
    """Clayton-coupled exponential pairs with light censoring."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    w = rng.random(n)
    # Clayton conditional inversion
    v = (u ** (-theta) * (w ** (-theta / (1 + theta)) - 1) + 1) ** (-1 / theta)
    t1 = -np.log(1 - u)
    t2 = -np.log(1 - v)
    c1 = rng.exponential(4.0, size=n)
    c2 = rng.exponential(4.0, size=n)
    return (
        np.minimum(t1, c1), (t1 <= c1).astype(float),
        np.minimum(t2, c2), (t2 <= c2).astype(float),
    )


def test_clyfr_recovers_theta_from_tau():
    hits = 0
    for seed in range(6):
        t1, e1, t2, e2 = _paired_survival(seed, theta=2.0)
        out = clayton_copula_frailty(t1, e1, t2, e2)
        # true tau = 2/(2+2) = 0.5; censoring attenuates the sample tau
        hits += abs(out["tau"] - 0.5) < 0.2
        assert np.all(out["joint_survival"] <= np.minimum(out["s1"], out["s2"]) + 1e-8)
    assert hits >= 5  # measured 6/6
    with pytest.raises(ValueError):
        clayton_copula_frailty([1.0] * 6, [1] * 6, [1.0] * 6, [1] * 6, theta=-1.0)


def test_copfr_family_choice():
    t1, e1, t2, e2 = _paired_survival(1, theta=2.0)
    cl = copula_frailty(t1, e1, t2, e2, family="clayton")
    gu = copula_frailty(t1, e1, t2, e2, family="gumbel")
    assert cl["tau_sample"] == pytest.approx(gu["tau_sample"])  # same data
    # each family's theta reproduces the sample tau through its own map
    assert copula_tau("clayton", cl["theta"]) == pytest.approx(cl["tau_sample"], abs=1e-6)
    assert copula_tau("gumbel", gu["theta"]) == pytest.approx(gu["tau_sample"], abs=1e-6)
    ind = copula_frailty(t1, e1, t2, e2, family="independence")
    assert ind["joint_survival"] == pytest.approx(ind["s1"] * ind["s2"])
    with pytest.raises(ValueError):
        copula_frailty(t1, e1, t2, e2, family="weibull")


def test_copod_flags_the_planted_outlier():
    rng = np.random.default_rng(0)
    from morie.fn.copod import copod

    X = rng.normal(size=(200, 4))
    X[0] = 8.0  # extreme in every dimension
    out = copod(X)
    assert int(np.argmax(out["scores"])) == 0
    assert out["scores"][0] > 2 * np.median(out["scores"])
    assert out["skewness"].size == 4
    # rank-based: rescaling any column must not change the ranking
    Y = X.copy()
    Y[:, 1] *= 1000.0
    assert np.argsort(copod(Y)["scores"])[-1] == 0
    with pytest.raises(ValueError):
        copod(np.array([[1.0, 2.0]]))


def _corr_data(seed=0, n=400, rho=0.7):
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, rho, 0.2], [rho, 1.0, 0.1], [0.2, 0.1, 1.0]])
    return rng.multivariate_normal(np.zeros(3), cov, size=n)


def test_zxcpg_recovers_the_correlation_matrix():
    from morie.fn.zxcpg import copula_gauss_sp

    X = _corr_data()
    out = copula_gauss_sp(X)
    assert out["correlation"][0, 1] == pytest.approx(0.7, abs=0.1)
    assert np.allclose(np.diag(out["correlation"]), 1.0)
    assert out["positive_definite"] is True
    assert out["pseudo_obs"].shape == (400, 3)
    assert np.all((out["pseudo_obs"] > 0) & (out["pseudo_obs"] < 1))
    with pytest.raises(ValueError):
        copula_gauss_sp(X[:, :1])


def test_zxcpc_marks_pairs_clayton_cannot_represent():
    from morie.fn.zxcpc import copula_clayton_sp

    rng = np.random.default_rng(1)
    cov = np.array([[1.0, 0.7, -0.6], [0.7, 1.0, -0.4], [-0.6, -0.4, 1.0]])
    X = rng.multivariate_normal(np.zeros(3), cov, size=400)
    out = copula_clayton_sp(X)
    assert out["theta_matrix"][0, 1] > 0  # positive pair fitted
    assert np.isnan(out["theta_matrix"][0, 2])  # negative pair marked, not clamped
    # the fitted theta reproduces that pair's tau
    assert copula_tau("clayton", out["theta_matrix"][0, 1]) == pytest.approx(
        out["tau_matrix"][0, 1], abs=1e-6
    )


def test_zxcpv_picks_the_hub_as_root():
    from morie.fn.zxcpv import copula_vine_sp

    rng = np.random.default_rng(2)
    # variable 1 is the hub: correlated with both others, which are not
    # correlated with each other
    z = rng.normal(size=400)
    X = np.column_stack([
        0.9 * z + 0.4 * rng.normal(size=400),
        z,
        0.9 * z + 0.4 * rng.normal(size=400),
    ])
    out = copula_vine_sp(X)
    assert out["root"] == 1
    assert len(out["tree1_edges"]) == 2
    assert all(fam in ("gumbel", "frank") for fam, _ in out["tree1_theta"])
    with pytest.raises(ValueError):
        copula_vine_sp(X[:3])
