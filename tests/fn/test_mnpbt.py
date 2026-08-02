"""mnpbt: multinomial probit choice probabilities.

Train, K. E. (2009), Discrete Choice Methods with Simulation, 2nd ed.,
Cambridge University Press, for the GHK simulator. The module previously
cited "Armstrong Ch 9"; that book has six chapters.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.mnpbt import multinomial_probit_spatial as mp


def test_mnpbt_probabilities_sum_to_one_per_observation():
    rng = np.random.default_rng(47)
    P = np.asarray(mp(rng.standard_normal((20, 4)), n_draws=4000, seed=1)["probs"])
    assert P.shape == (20, 4)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P >= 0.0)


def test_mnpbt_equal_utilities_give_equal_probabilities():
    """With no deterministic advantage the choice is decided by noise alone,
    so each of k alternatives is chosen 1/k of the time."""
    P = np.asarray(mp(np.zeros((1, 4)), n_draws=40_000, seed=2)["probs"])
    assert P[0] == pytest.approx(np.full(4, 0.25), abs=0.02)


def test_mnpbt_higher_utility_means_higher_probability():
    """The ordering of probabilities must follow the ordering of utilities."""
    P = np.asarray(mp(np.array([[0.0, 1.0, 2.0]]), n_draws=40_000, seed=3)["probs"])[0]
    assert P[0] < P[1] < P[2]


def test_mnpbt_dominant_alternative_approaches_certainty():
    P = np.asarray(mp(np.array([[0.0, 0.0, 12.0]]), n_draws=20_000, seed=4)["probs"])[0]
    assert P[2] > 0.99


def test_mnpbt_is_invariant_to_adding_a_constant_to_every_utility():
    """Only utility DIFFERENCES identify a discrete-choice model; shifting all
    alternatives by the same amount must change nothing."""
    U = np.array([[0.0, 1.0, 2.0]])
    a = np.asarray(mp(U, n_draws=20_000, seed=5)["probs"])
    b = np.asarray(mp(U + 7.5, n_draws=20_000, seed=5)["probs"])
    assert a == pytest.approx(b, abs=1e-12)


def test_mnpbt_max_alt_is_the_argmax_of_the_probabilities():
    U = np.array([[0.0, 3.0, 1.0], [4.0, 0.0, 0.0]])
    r = mp(U, n_draws=20_000, seed=6)
    assert np.asarray(r["max_alt"]).tolist() == [1, 0]


def test_mnpbt_is_reproducible_for_a_fixed_seed():
    U = np.array([[0.0, 1.0, 2.0]])
    a = np.asarray(mp(U, n_draws=5000, seed=99)["probs"])
    b = np.asarray(mp(U, n_draws=5000, seed=99)["probs"])
    assert a == pytest.approx(b, abs=0.0)


def test_mnpbt_reports_shape_and_accepts_one_observation():
    r = mp(np.array([1.0, 2.0, 3.0]), n_draws=2000, seed=7)
    assert r["n_obs"] == 1 and r["n_alt"] == 3
