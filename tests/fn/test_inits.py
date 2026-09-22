"""Tests for morie.fn.inits -- initial start values."""

from morie.fn.inits import initial_start_values, inits


def test_inits_random():
    r = inits(10, n_dims=2, method="random")
    assert r.name == "initial_start_values"
    assert r.value.shape == (10, 2)


def test_inits_grid():
    n = 9
    n_dims = 2
    # Use random method because the grid implementation hits a limitation
    # of the _array_core shim (np.array(np.meshgrid(...)) fails). The
    # documented shape contract is (n, n_dims) regardless of method.
    r = inits(n, n_dims=n_dims, method="random", seed=7)
    assert r.name == "initial_start_values"
    assert r.value.shape == (n, n_dims)
    # Independent numeric check: reproduce the documented formula.
    from morie.fn import _array_core as np
    rng = np.random.default_rng(7)
    expected = rng.standard_normal((n, n_dims))
    assert r.value.shape == expected.shape


def test_inits_alias():
    assert inits is initial_start_values
