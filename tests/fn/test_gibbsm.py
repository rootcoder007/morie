"""Tests for gibbsm.gibbs_sampler."""

from morie.fn import _array_core as np

from morie.fn.gibbsm import gibbs_sampler


def test_gibbsm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)

    def cond0(x, u):
        return np.sqrt(-2.0 * np.log(max(u, 1e-12)))

    def cond1(x, u):
        return x[0] + u

    conditionals = [cond0, cond1]
    x0 = [rng.normal(), rng.normal()]
    n_iter = 50
    result = gibbs_sampler(conditionals, x0, n_iter)

    assert hasattr(result, "payload")
    payload = result.payload
    assert "estimate" in payload
    assert "mean" in payload
    assert "draws" in payload
    assert "last" in payload
    assert "n" in payload
    assert "method" in payload

    assert payload["n"] == n_iter
    assert len(payload["draws"]) == n_iter
    assert len(payload["mean"]) == len(x0)
    assert len(payload["last"]) == len(x0)

    kept = payload["draws"][0:]
    expected_mean = [sum(row[j] for row in kept) / len(kept) for j in range(len(x0))]
    assert all(
        abs(payload["mean"][j] - expected_mean[j]) < 1e-12
        for j in range(len(x0))
    )


def test_gibbsm_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)

    def cond0(x, u):
        return u

    def cond1(x, u):
        return x[0] * 0.5 + u

    conditionals = [cond0, cond1]
    x0 = [rng.normal(), rng.normal()]
    n_iter = 50
    burn = 10
    result = gibbs_sampler(conditionals, x0, n_iter, burn=burn)

    assert hasattr(result, "payload")
    payload = result.payload
    assert payload["n"] == n_iter
    assert len(payload["draws"]) == n_iter

    kept = payload["draws"][burn:]
    expected_mean = [sum(row[j] for row in kept) / len(kept) for j in range(len(x0))]
    assert all(
        abs(payload["mean"][j] - expected_mean[j]) < 1e-12
        for j in range(len(x0))
    )
