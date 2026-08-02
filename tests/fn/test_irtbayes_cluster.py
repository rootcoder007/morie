"""IRT/Bayesian + survey cluster: ambtc, bayam, irtdq, irtid, foldp,
plpol, mcmpp, pscli, hsirt, emtxt, bymds, bmdul, chopit."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ambtc import am_bootstrap_se
from morie.fn.bayam import bayesian_am_scaling
from morie.fn.bmdul import bayesian_mds_unfolding
from morie.fn.bymds import bayesian_mds
from morie.fn.chopit import chopit_vignette
from morie.fn.emtxt import em_irt_text
from morie.fn.foldp import folding_problem
from morie.fn.hsirt import heteroskedastic_irt
from morie.fn.irtdq import irt_quadratic_utility
from morie.fn.irtid import irt_identification_constraints
from morie.fn.mcmpp import mcmcpack_irt
from morie.fn.plpol import plot_spatial
from morie.fn.pscli import pscl_ideal
from morie.fn.pscrc import pscl_rollcall


def _am_data(seed=0, n=80, q=6):
    """A-M world: true stimuli, respondent-specific shift and stretch."""
    rng = np.random.default_rng(seed)
    s = np.linspace(-1.5, 1.5, q)
    a = rng.normal(scale=0.5, size=n)
    b = rng.uniform(0.5, 1.5, size=n)
    Z = a[:, None] + b[:, None] * s[None, :] + rng.normal(scale=0.15, size=(n, q))
    return Z, s


def _votes(seed=0, n=40, q=60):
    """Probit IRT world with known ideal-point ORDER."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-2, 2, n)
    beta = rng.normal(scale=1.0, size=q)
    alpha = rng.normal(scale=0.5, size=q)
    from morie.fn._stats_core import norm

    P = norm.cdf(x[:, None] * beta[None, :] - alpha[None, :])
    return (rng.random((n, q)) < P).astype(float), x


def test_ambtc_bootstrap_se():
    Z, s = _am_data()
    out = am_bootstrap_se(Z, B=40, seed=0)
    assert out["se"].shape == (6,)
    assert np.all(out["se"] > 0)
    assert np.all(out["se"] < 0.5)  # measured ~0.01-0.05 at n=80
    # the full-sample zhat tracks the true stimulus order
    assert np.corrcoef(out["zhat"], s)[0, 1] > 0.99
    with pytest.raises(ValueError):
        am_bootstrap_se(Z, B=5)


def test_bayam_recovers_stimuli():
    Z, s = _am_data(1)
    out = bayesian_am_scaling(Z, n_iter=600, burnin=200, seed=0)
    s_norm = (s - s.mean()) / s.std()
    est = out["stimuli"]
    if np.corrcoef(est, s_norm)[0, 1] < 0:  # reflection is unidentified
        est = -est
    assert np.corrcoef(est, s_norm)[0, 1] > 0.99
    assert out["stimuli_ci"].shape == (2, 6)
    with pytest.raises(ValueError):
        bayesian_am_scaling(Z, n_iter=100, burnin=200)


def test_irtdq_probit_geometry():
    # at the midpoint the probability is exactly one half
    out = irt_quadratic_utility(0.5, yea_position=0.0, nay_position=1.0)
    assert out["p_yea"] == pytest.approx(0.5)
    assert out["midpoint"] == pytest.approx(0.5)
    # yea below nay: ideal points below the midpoint favour yea
    low = irt_quadratic_utility(-1.0, 0.0, 1.0)["p_yea"]
    assert low > 0.9
    with pytest.raises(ValueError):
        irt_quadratic_utility(0.0, 1.0, 1.0)  # coincident outcomes


def test_irtid_constraints():
    x = np.array([3.0, 5.0, 7.0, 9.0])
    out = irt_identification_constraints(x, polarity_idx=3)
    assert out["x"].mean() == pytest.approx(0.0, abs=1e-12)
    assert out["x"].std() == pytest.approx(1.0, abs=1e-12)
    assert out["x"][3] < 0 and out["reflected"] is True
    with pytest.raises(ValueError):
        irt_identification_constraints(x, polarity_idx=0, pivot_idx=1)  # same side
    with pytest.raises(ValueError):
        irt_identification_constraints(np.ones(4))  # constant


def test_foldp_single_peaked():
    # perfect unfolding data is single-peaked; shuffled stimulus order is not
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 7)
    T = 5.0 - (x[:, None] - y[None, :]) ** 2
    out = folding_problem(T)
    assert out["single_peaked_share"] == pytest.approx(1.0)
    bad = folding_problem(T, stimulus_order=[3, 0, 5, 1, 6, 2, 4])
    assert bad["single_peaked_share"] < 0.5
    with pytest.raises(ValueError):
        folding_problem(T, stimulus_order=[0, 0, 1, 2, 3, 4, 5])


def test_plpol_data_layer():
    pts = np.array([[-1.0, 0.0], [1.0, 0.5], [0.8, -0.2]])
    out = plot_spatial(pts, party_labels=["D", "R", "R"])
    assert out["centroids"]["R"] == pytest.approx([0.9, 0.15])
    assert out["xlim"][0] < -1.0 < 1.0 < out["xlim"][1]
    one_d = plot_spatial([0.0, 1.0, 2.0])
    assert one_d["coords"].shape == (3, 2)
    with pytest.raises(ValueError):
        plot_spatial(pts, party_labels=["D"])


def test_mcmpp_recovers_order():
    V, x_true = _votes()
    out = mcmcpack_irt(V, n_iter=400, burnin=150, seed=0, polarity_idx=0)
    est = out["ideal_points"]
    r = np.corrcoef(est, x_true)[0, 1]
    assert r > 0.9  # measured ~0.97 at 60 items
    assert est[0] < 0  # polarity respected
    assert out["ideal_ci"].shape == (2, V.shape[0])
    with pytest.raises(ValueError):
        mcmcpack_irt(V * 2)  # non-binary
    with pytest.raises(ValueError):
        mcmcpack_irt(V, n_iter=100, burnin=200)


def test_pscli_pipeline():
    V, x_true = _votes(1)
    # add two unanimous roll calls that the screen must drop
    V2 = np.column_stack([V, np.ones(V.shape[0]), np.zeros(V.shape[0])])
    rc = pscl_rollcall(V2, lop=0.025)
    out = pscl_ideal(rc, n_iter=300, burnin=100, seed=0, polarity_idx=0)
    assert out["n_rollcalls_dropped"] == 2
    r = np.corrcoef(out["ideal_points"], x_true)[0, 1]
    assert abs(r) > 0.9
    with pytest.raises(ValueError):
        pscl_ideal({"votes": V2})  # missing 'keep'


def test_hsirt_flags_the_noisy_voter():
    rng = np.random.default_rng(2)
    n, q = 30, 80
    x = np.linspace(-2, 2, n)
    beta = rng.normal(scale=1.2, size=q)
    alpha = rng.normal(scale=0.5, size=q)
    from morie.fn._stats_core import norm

    psi_true = np.ones(n)
    psi_true[0] = 4.0  # one unpredictable voter
    P = norm.cdf((x[:, None] * beta[None, :] - alpha[None, :]) / psi_true[:, None])
    V = (rng.random((n, q)) < P).astype(float)
    out = heteroskedastic_irt(V, x, item_params=(alpha, beta), max_iter=10)
    assert np.argmax(out["psi"]) == 0  # the noisy voter has the largest scale
    assert out["psi"][0] > 1.5 * np.median(out["psi"])
    assert np.exp(np.mean(np.log(out["psi"]))) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError):
        heteroskedastic_irt(V, x[:5])


def test_emtxt_wordfish_recovers_positions():
    rng = np.random.default_rng(3)
    n, k = 10, 60
    theta = np.linspace(-1.5, 1.5, n)
    psi = rng.normal(1.0, 0.3, size=k)
    beta = rng.normal(0.0, 0.8, size=k)
    lam = np.exp(1.0 + psi[None, :] + beta[None, :] * theta[:, None])
    Y = rng.poisson(lam)
    out = em_irt_text(Y, polarity=(0, n - 1))
    r = np.corrcoef(out["theta"], theta)[0, 1]
    assert r > 0.95  # measured ~0.99
    assert out["theta"][0] < out["theta"][-1]  # polarity respected
    with pytest.raises(ValueError):
        em_irt_text(-Y)


def test_bymds_posterior_tracks_torgerson():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(8, 2))
    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt((diff**2).sum(axis=2))
    out = bayesian_mds(D, n_dims=2, n_iter=400, burnin=150, seed=0, step=0.03)
    est = out["coordinates"]
    de = np.sqrt(((est[:, None, :] - est[None, :, :]) ** 2).sum(axis=2))
    iu = np.triu_indices(8, 1)
    # measured 0.9887 at 400 sweeps; the threshold leaves seed room
    assert np.corrcoef(de[iu], D[iu])[0, 1] > 0.97
    assert np.all(out["ci_radius"] > 0)
    assert 0 < out["acceptance"] < 1
    with pytest.raises(ValueError):
        bayesian_mds(D, n_iter=100, burnin=200)


def test_bmdul_unfolding_recovers_geometry():
    rng = np.random.default_rng(5)
    n, q = 25, 6
    Xt = rng.uniform(-1, 1, size=(n, 1))
    Yt = np.linspace(-1, 1, q)[:, None]
    T = 4.0 - ((Xt[:, None, 0] - Yt[None, :, 0]) ** 2) + rng.normal(scale=0.1, size=(n, q))
    out = bayesian_mds_unfolding(T, n_dims=1, n_iter=500, burnin=200, seed=0, step=0.05)
    est = out["stimuli"][:, 0]
    r = np.corrcoef(est, Yt[:, 0])[0, 1]
    assert abs(r) > 0.95  # sign is unidentified
    assert 0 < out["acceptance"] < 1
    with pytest.raises(ValueError):
        bayesian_mds_unfolding(T, n_dims=0)


def test_chopit_recovers_dif_shift():
    rng = np.random.default_rng(6)
    n = 400
    grp = np.repeat(["a", "b"], n // 2)
    shift = np.where(grp == "a", 0.0, 0.8)  # group b uses harsher thresholds
    taus = np.array([-0.5, 0.5])
    mu_v = np.array([-0.8, 0.6])  # two vignettes, same for everyone

    def rate(latent, sh):
        # per-respondent shifted thresholds: category = 1 + #{k: latent > tau_k + sh_i}
        return 1 + (latent[:, None] > (taus[None, :] + sh[:, None])).sum(axis=1)

    Vg = np.column_stack(
        [rate(mu_v[j] + rng.normal(scale=1.0, size=n), shift) for j in range(2)]
    )
    self_lat = rng.normal(scale=1.0, size=n)  # same true distribution in both groups
    y = rate(self_lat, shift)
    out = chopit_vignette(y, Vg, group=grp, n_categories=3)
    assert out["dif_shift"]["a"] == pytest.approx(0.0)
    assert out["dif_shift"]["b"] == pytest.approx(0.8, abs=0.3)
    # both groups share the same true latent distribution (mean 0):
    # the corrected latent means must agree, though the naive ordinal
    # means differ by construction (measured naive gap ~0.46)
    assert abs(out["naive_means"]["a"] - out["naive_means"]["b"]) > 0.2
    assert out["corrected_means"]["a"] == pytest.approx(
        out["corrected_means"]["b"], abs=0.25
    )
    # location anchor: mean vignette level = 0, true self mean sits at
    # -mean(mu_v) = 0.1 on that scale
    assert out["corrected_means"]["a"] == pytest.approx(0.1, abs=0.3)
    with pytest.raises(ValueError):
        chopit_vignette(y, Vg[:10], group=grp)
