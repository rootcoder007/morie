"""RNA velocity from the full splicing kinetics."""
import importlib
import math

import pytest

S = importlib.import_module("morie.fn.scvelo")
A, B, G = 5.0, 1.2, 0.6


def rk4(T, alpha=A, beta=B, gamma=G, u0=0.0, s0=0.0, n=8000):
    u, s, h = u0, s0, T / n

    def f(u, s):
        return (alpha - beta * u, beta * u - gamma * s)

    for _ in range(n):
        k1 = f(u, s)
        k2 = f(u + h / 2 * k1[0], s + h / 2 * k1[1])
        k3 = f(u + h / 2 * k2[0], s + h / 2 * k2[1])
        k4 = f(u + h * k3[0], s + h * k3[1])
        u += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        s += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return u, s


@pytest.mark.parametrize("T", [0.25, 1.0, 3.0])
def test_closed_form_solves_the_odes(T):
    cf = S.solve_kinetics(T, A, B, G)
    u, s = rk4(T)
    assert abs(cf["u"] - u) < 1e-9
    assert abs(cf["s"] - s) < 1e-9


def test_closed_form_holds_from_a_switch_state():
    cf = S.solve_kinetics(4.0, A, B, G, u0=2.0, s0=3.0)
    u, s = rk4(4.0, u0=2.0, s0=3.0)
    assert abs(cf["u"] - u) < 1e-9
    assert abs(cf["s"] - s) < 1e-9


def test_equal_rates_are_a_removable_singularity():
    lim = S.solve_kinetics(3.0, A, 0.9, 0.9)
    assert abs(lim["s"] - rk4(3.0, beta=0.9, gamma=0.9)[1]) < 1e-9


def test_steady_state_is_alpha_over_each_rate():
    far = S.solve_kinetics(80.0, A, B, G)
    assert abs(far["u"] - A / B) < 1e-9
    assert abs(far["s"] - A / G) < 1e-9
    assert abs(S.velocity(A / B, A / G, B, G)) < 1e-12


def test_velocity_signs_track_induction_and_repression():
    sim = S.simulate_gene(A, B, G, 4.0, [0.5, 2.0, 3.9, 4.5, 8.0])
    on = [o for o in sim["observations"] if o["state"] == "on"]
    off = [o for o in sim["observations"] if o["state"] == "off"]
    assert on and off
    assert all(o["velocity"] > 0 for o in on)
    assert all(o["velocity"] < 0 for o in off)


@pytest.mark.parametrize("bad", [(-1.0, A, B, G), (1.0, A, 0.0, G),
                                 (1.0, A, B, 0.0), (1.0, -1.0, B, G)])
def test_impossible_kinetics_are_refused(bad):
    with pytest.raises(ValueError):
        S.solve_kinetics(*bad)


def test_steady_state_model_fits_the_u_on_s_slope():
    obs = S.simulate_gene(A, B, G, 5.0,
                          [0.2 * k for k in range(1, 60)])["observations"]
    ss = S.steady_state_velocity([o["u"] for o in obs],
                                 [o["s"] for o in obs])
    assert abs(ss["gamma_over_beta"] - G / B) < 0.1 * (G / B)
    assert "steady states observed" in ss["assumptions"]


def test_steady_state_model_needs_more_than_two_cells():
    with pytest.raises(ValueError):
        S.steady_state_velocity([1.0, 2.0], [1.0, 2.0])


def _transient():
    sim = S.simulate_gene(A, B, G, 5.0,
                          [0.1 * k for k in range(1, 18)])
    return ([o["u"] for o in sim["observations"]],
            [o["s"] for o in sim["observations"]])


def test_em_never_increases_the_residual():
    u, s = _transient()
    fit = S.dynamical_fit(u, s, alpha0=4.0, beta0=1.0, gamma0=0.5,
                          t_switch0=4.0, n_iter=12, grid=80)
    h = fit["rss_history"]
    assert all(h[i] >= h[i + 1] - 1e-9 for i in range(len(h) - 1))


def test_dynamical_model_fits_without_a_steady_state():
    u, s = _transient()
    fit = S.dynamical_fit(u, s, alpha0=4.0, beta0=1.0, gamma0=0.5,
                          t_switch0=4.0, n_iter=12, grid=80)
    assert max(s) < 0.6 * (A / G)
    assert fit["rss"] < 0.5
    assert len(fit["latent"]) == len(u)
    assert all(x["state"] in S.STATES for x in fit["latent"])


def test_genes_couple_into_one_shared_clock():
    u, s = _transient()
    fit = S.dynamical_fit(u, s, alpha0=4.0, beta0=1.0, gamma0=0.5,
                          t_switch0=4.0, n_iter=8, grid=60)
    lt = S.latent_time([fit, fit])
    assert lt["n_genes"] == 2
    assert len(lt["latent_time"]) == len(u)


def test_an_empty_gene_set_is_refused():
    with pytest.raises(ValueError):
        S.latent_time([])


def test_the_alias_resolves_to_the_dynamical_model():
    assert S.rna_velocity is S.dynamical_fit
