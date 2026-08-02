"""Panel-data deconvolution and first-passage times (Horowitz Sec. 5.2)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.hrzfneps import horowitz_fn_eps_fn_U
from morie.fn.hrzfnu import horowitz_smoothed_fU
from morie.fn.hrzfpt import horowitz_first_passage_time
from morie.fn.hrzpanel import horowitz_panel_deconvolution


def _panel(n=800, T=4, seed=0, su=1.0, se=0.5):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, T, 2))
    beta = np.array([1.0, -0.5])
    U = rng.standard_normal(n) * su
    eps = rng.standard_normal((n, T)) * se     # symmetric, as assumed
    Y = X @ beta + U[:, None] + eps
    return Y, X, beta, su, se


def test_panel_deconvolution_separates_the_two_variance_components():
    Y, X, beta, su, se = _panel()
    out = horowitz_panel_deconvolution(Y, X, beta)
    # the recovered densities should have roughly the right spread:
    # integrate z^2 f(z) and compare with the true variances
    def var_of(g, f):
        m = np.trapezoid(f, g)
        mu = np.trapezoid(g * f, g) / m
        return np.trapezoid((g - mu) ** 2 * f, g) / m
    vU = var_of(out["grid_u"], np.clip(out["f_U"], 0, None))
    vE = var_of(out["grid_z"], np.clip(out["f_eps"], 0, None))
    assert abs(np.sqrt(vU) - su) < 0.6
    # f_eps is estimated from eta, the DIFFERENCE of two epsilons,
    # so its spread is the eps scale, not the eta scale
    assert abs(np.sqrt(vE) - se * np.sqrt(2)) < 0.6
    assert out["symmetry_required"] is True
    assert out["asymptotics_in"] == "n with T fixed"
    assert out["T"] == 4


def test_the_differenced_residual_removes_the_individual_effect():
    from morie.fn._hrz_paneldec import panel_residuals
    # a huge individual effect must not touch eta at all
    Y, X, beta, _, _ = _panel(n=300, su=1.0)
    Y2 = Y + np.arange(300)[:, None] * 100.0   # enormous extra U_j
    W1, eta1 = panel_residuals(Y, X, beta)
    W2, eta2 = panel_residuals(Y2, X, beta)
    assert np.allclose(eta1, eta2)             # eta is untouched
    assert not np.allclose(W1, W2)             # W is not


def test_smoothed_fU_integrates_to_about_one_and_reports_its_cutoff():
    Y, X, beta, _, _ = _panel(n=400)
    out = horowitz_smoothed_fU(Y, X, beta)
    mass = np.trapezoid(out["f_U"], out["grid"])
    assert 0.5 < mass < 1.2   # the grid is trimmed to the 5-95% range
    assert out["cutoff"] == pytest.approx(1.0 / out["nu_U"])
    assert out["regularisation_required"] is True
    with pytest.raises(ValueError):
        horowitz_smoothed_fU(Y, X, beta, nu_U=-1.0)


def test_the_two_estimators_carry_separate_bandwidths():
    Y, X, beta, _, _ = _panel(n=400)
    out = horowitz_fn_eps_fn_U(Y, X, beta, nu_U=0.4, nu_eps=0.9)
    assert out["nu_U"] == 0.4 and out["nu_eps"] == 0.9
    # only f_U divides by |psi_eta|^{1/2}
    assert out["f_U_requires_division"] is True
    assert out["f_eps_requires_division"] is False
    # changing nu_eps must move f_eps and leave f_U alone
    other = horowitz_fn_eps_fn_U(Y, X, beta, nu_U=0.4, nu_eps=0.5)
    assert np.allclose(out["f_U"], other["f_U"])
    assert not np.allclose(out["f_eps"], other["f_eps"])


def test_panel_deconvolution_validates_shapes():
    Y, X, beta, _, _ = _panel(n=50, T=3)
    with pytest.raises(ValueError):
        horowitz_panel_deconvolution(Y, X, np.array([1.0]))
    with pytest.raises(ValueError):
        horowitz_panel_deconvolution(Y[:, :1], X[:, :1, :], beta)  # T = 1
    with pytest.raises(ValueError):
        horowitz_panel_deconvolution(Y[:5], X[:5], beta)           # n = 5


def test_first_passage_probability_falls_with_the_horizon():
    gu = np.linspace(-5, 5, 201)
    gz = np.linspace(-5, 5, 201)
    fu = np.exp(-0.5 * gu**2) / np.sqrt(2 * np.pi)
    fe = np.exp(-0.5 * (gz / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
    beta = np.array([1.0, -0.5])
    X = np.zeros((8, 2))
    p = [horowitz_first_passage_time(th, 0.0, 1.0, X, beta, fu, gu, fe, gz)
         ["probability"] for th in (2, 4, 8)]
    # surviving longer is never more likely
    assert p[0] > p[1] > p[2]
    assert all(0.0 <= v <= 1.0 for v in p)


def test_first_passage_beats_the_naive_independent_product():
    # Conditional on U the periods are independent; unconditionally
    # they are NOT, because they share U. Multiplying marginal
    # probabilities therefore understates survival -- the shared
    # effect makes the periods positively dependent.
    gu = np.linspace(-6, 6, 301)
    gz = np.linspace(-6, 6, 301)
    fu = np.exp(-0.5 * (gu / 1.5) ** 2) / (1.5 * np.sqrt(2 * np.pi))
    fe = np.exp(-0.5 * (gz / 0.5) ** 2) / (0.5 * np.sqrt(2 * np.pi))
    beta = np.array([1.0, -0.5])
    X = np.zeros((6, 2))
    out = horowitz_first_passage_time(6, 0.0, 1.0, X, beta, fu, gu, fe, gz)
    # marginal P(Y_k <= y*) with U integrated out per period
    from scipy import stats
    marg = float(stats.norm.cdf(1.0, 0.0, np.sqrt(1.5**2 + 0.5**2)))
    assert out["probability"] > marg ** 5
    assert out["periods_conditionally_independent"] is True
    assert out["periods_marginally_independent"] is False


def test_first_passage_validates_its_inputs():
    gu = np.linspace(-4, 4, 101)
    gz = np.linspace(-4, 4, 101)
    fu = np.exp(-0.5 * gu**2)
    fe = np.exp(-0.5 * gz**2)
    beta = np.array([1.0, -0.5])
    X = np.zeros((5, 2))
    with pytest.raises(ValueError):
        horowitz_first_passage_time(1, 0.0, 1.0, X, beta, fu, gu, fe, gz)
    with pytest.raises(ValueError):
        horowitz_first_passage_time(9, 0.0, 1.0, X, beta, fu, gu, fe, gz)
    with pytest.raises(ValueError):
        horowitz_first_passage_time(3, 0.0, 1.0, X, beta, -fu, gu, fe, gz)
    with pytest.raises(ValueError):
        horowitz_first_passage_time(3, 0.0, 1.0, X, beta, fu[:10], gu, fe, gz)
