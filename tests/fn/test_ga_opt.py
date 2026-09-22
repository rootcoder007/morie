"""Tests for ga_opt.genetic_algorithm."""

from morie.fn import _array_core as np

from morie.fn.ga_opt import genetic_algorithm


def test_ga_opt_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    # f must be callable (e.g. a lambda/function)
    f = lambda x: float(sum(xi * xi for xi in x))
    # population is a sequence of candidate vectors
    population = [rng.normal(0.0, 1.0, 4).tolist() for _ in range(10)]
    generations = 5
    result = genetic_algorithm(f, population, generations)
    # The implementation returns a RichResult; accept either dict-like or the
    # object itself so the test stays robust.
    assert hasattr(result, "payload") or isinstance(result, dict)
    payload = result.payload if hasattr(result, "payload") else result
    # Documented keys in the payload
    for key in ("estimate", "best", "best_fitness", "best_path",
                "generations", "n", "method"):
        assert key in payload
    # best_path should have ng+1 entries (one per generation plus the final one)
    assert len(payload["best_path"]) == generations + 1
    # n should equal the number of individuals passed in
    assert payload["n"] == len(population)
    assert payload["generations"] == generations
    # estimate and best_fitness must agree, and best must have the right length
    assert payload["estimate"] == payload["best_fitness"]
    assert len(payload["best"]) == len(population[0])


def test_ga_opt_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    f = lambda x: float(sum(xi * xi for xi in x))
    population = [rng.normal(0.0, 1.0, 3).tolist() for _ in range(6)]
    # Calling with the default generations should also work
    result = genetic_algorithm(f, population)
    assert hasattr(result, "payload") or isinstance(result, dict)
    payload = result.payload if hasattr(result, "payload") else result
    assert "best_path" in payload
    assert "best" in payload
