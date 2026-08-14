"""hiatus -- many-strain dynamics. Source: Gog, J. R. & Grenfell,
B. T. (2002) PNAS 99(26), 17209-17214, doi:10.1073/pnas.252512799."""
import pytest

from morie.fn.hiatus import (basic_reproduction_numbers, derivatives,
                             endemic_equilibrium,
                             linear_strain_space, simulate)

MU = 1.0 / (50.0 * 365.0)
NU = 0.2


def test_r0_is_beta_over_nu_plus_mu():
    r = basic_reproduction_numbers([0.6], NU, MU)[0]
    assert r == pytest.approx(0.6 / (NU + MU))


def test_equilibrium_susceptibles_are_one_over_r0():
    eq = endemic_equilibrium([0.6], NU, MU)
    assert eq["S"] == pytest.approx(1.0 / eq["R0"], abs=1e-15)


def test_the_equilibrium_is_a_fixed_point():
    eq = endemic_equilibrium([0.6], NU, MU)
    dS, dI = derivatives([eq["S"]], [eq["I"]], [0.6], NU, MU, [[1.0]])
    assert abs(dS[0]) < 1e-18
    assert abs(dI[0]) < 1e-18


def test_r0_below_one_has_no_endemic_state():
    eq = endemic_equilibrium([0.05], NU, MU)
    assert eq["R0"] < 1.0
    assert eq["I"] == 0.0


def test_derivatives_match_the_printed_equations():
    dS, dI = derivatives([0.5, 0.4], [0.01, 0.02], [0.6, 0.5], 0.2,
                         0.01, [[1.0, 0.3], [0.3, 1.0]])
    assert dI[0] == pytest.approx(
        0.6 * 0.5 * 0.01 - 0.2 * 0.01 - 0.01 * 0.01, abs=1e-15)
    assert dS[0] == pytest.approx(
        0.01 - (0.6 * 0.5 * 0.01 + 0.5 * 0.5 * 0.3 * 0.02)
        - 0.01 * 0.5, abs=1e-15)


def test_no_cross_immunity_decouples_the_strains_exactly():
    beta = [0.6, 0.5]
    sig = [[1.0, 0.0], [0.0, 1.0]]
    joint = simulate(beta, NU, MU, sig, S0=[0.9, 0.8],
                     I0=[1e-3, 2e-3], t_end=200.0, dt=0.1)
    for i in range(2):
        solo = simulate([beta[i]], NU, MU, [[1.0]],
                        S0=[[0.9, 0.8][i]], I0=[[1e-3, 2e-3][i]],
                        t_end=200.0, dt=0.1)
        assert joint["S"][i] == pytest.approx(solo["S"][0], abs=1e-14)
        assert joint["I"][i] == pytest.approx(solo["I"][0], abs=1e-14)


def test_cross_immunity_does_change_the_trajectory():
    beta = [0.6, 0.5]
    a = simulate(beta, NU, MU, [[1.0, 0.0], [0.0, 1.0]],
                 S0=[0.9, 0.8], I0=[1e-3, 2e-3], t_end=200.0, dt=0.1)
    b = simulate(beta, NU, MU, [[1.0, 0.6], [0.6, 1.0]],
                 S0=[0.9, 0.8], I0=[1e-3, 2e-3], t_end=200.0, dt=0.1)
    assert abs(a["S"][0] - b["S"][0]) > 1e-4


def test_the_variable_count_is_linear_in_the_strain_count():
    r = simulate([0.5] * 8, NU, MU, linear_strain_space(8),
                 t_end=50.0, dt=0.5)
    assert r["n_variables"] == 16
    assert "256" in r["n_variables_history_based"]


def test_susceptibles_stay_in_the_unit_interval():
    r = simulate([0.8, 0.6], NU, MU, [[1.0, 0.5], [0.5, 1.0]],
                 t_end=500.0, dt=0.2)
    assert all(0.0 <= v <= 1.0 for v in r["S"])
    assert all(v >= 0.0 for v in r["I"])


def test_strain_space_has_a_unit_diagonal():
    sp = linear_strain_space(5)
    assert all(sp[i][i] == pytest.approx(1.0) for i in range(5))


def test_strain_space_decays_with_distance_and_is_symmetric():
    sp = linear_strain_space(5, width=1.5)
    assert sp[0][1] > sp[0][2] > sp[0][3]
    assert all(sp[i][j] == pytest.approx(sp[j][i])
               for i in range(5) for j in range(5))


def test_a_wider_kernel_gives_more_cross_immunity():
    a = linear_strain_space(5, width=1.0)[0][2]
    b = linear_strain_space(5, width=3.0)[0][2]
    assert b > a


def test_a_sigma_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError):
        derivatives([0.5], [0.01], [0.6], 0.2, 0.01, [[1.5]])


def test_a_non_square_sigma_is_refused():
    with pytest.raises(ValueError):
        derivatives([0.5, 0.4], [0.01, 0.02], [0.6, 0.5], 0.2, 0.01,
                    [[1.0]])


def test_a_non_positive_transmission_rate_is_refused():
    with pytest.raises(ValueError):
        simulate([0.0], NU, MU, [[1.0]])


def test_a_mutation_rate_outside_the_range_is_refused():
    with pytest.raises(ValueError):
        simulate([0.5], NU, MU, [[1.0]], mutation=1.0)


def test_a_non_positive_dt_is_refused():
    with pytest.raises(ValueError):
        simulate([0.5], NU, MU, [[1.0]], dt=-0.1)


def test_a_mismatched_initial_state_is_refused():
    with pytest.raises(ValueError):
        simulate([0.5, 0.4], NU, MU, [[1.0, 0.0], [0.0, 1.0]],
                 S0=[0.9])
