"""Tests for epsig1.em_algorithm."""

from morie.fn import _array_core as np

from morie.fn.epsig1 import em_algorithm


def _constant_log_lik(value):
    """Return a log_lik callable that ignores its argument."""
    def _ll(theta):
        return float(value)
    return _ll


def _identity_Q():
    """Return a Q callable that returns its input (a list) unchanged."""
    def _q(theta):
        return [float(v) for v in theta]
    return _q


def test_epsig1_basic():
    """Test basic functionality with a small number of EM steps."""
    rng = np.random.default_rng(42)
    n = 4
    x0 = [float(v) for v in rng.normal(0.0, 1.0, n)]
    log_lik = _constant_log_lik(0.5)
    Q = _identity_Q()
    steps = 5

    result = em_algorithm(log_lik, Q, x0, steps)

    assert isinstance(result, dict)
    assert "theta" in result
    assert "loglik" in result
    assert "trace" in result
    assert "increments" in result
    assert "min_increment" in result
    assert "monotone" in result
    assert "steps" in result
    assert "method" in result

    # theta must equal x0 because Q is the identity
    assert [float(v) for v in result["theta"]] == [float(v) for v in x0]

    # steps must be the int we passed in
    assert int(result["steps"]) == steps

    # log-likelihood is constant 0.5 for every iterate
    expected_loglik = 0.5
    assert float(result["loglik"]) == expected_loglik

    # trace has steps+1 entries, all equal to the constant log-likelihood
    trace = [float(v) for v in result["trace"]]
    assert len(trace) == steps + 1
    for v in trace:
        assert v == expected_loglik

    # increments are all zero, min_increment is 0, monotone is True
    inc = [trace[i + 1] - trace[i] for i in range(len(trace) - 1)]
    assert inc == [0.0] * steps
    assert float(result["min_increment"]) == 0.0
    assert bool(result["monotone"]) is True


def test_epsig1_edge():
    """Edge case: zero steps should still produce a valid audit record."""
    rng = np.random.default_rng(42)
    n = 3
    x0 = [float(v) for v in rng.normal(0.0, 1.0, n)]
    log_lik = _constant_log_lik(-1.25)
    Q = _identity_Q()
    steps = 0

    result = em_algorithm(log_lik, Q, x0, steps)

    assert isinstance(result, dict)
    assert int(result["steps"]) == 0

    # With zero iterations, theta is the starting point
    assert [float(v) for v in result["theta"]] == [float(v) for v in x0]

    # trace contains only the initial log-likelihood; no increments
    trace = [float(v) for v in result["trace"]]
    assert len(trace) == 1
    assert trace[0] == -1.25
    assert float(result["loglik"]) == -1.25

    inc = [float(v) for v in result["increments"]]
    assert inc == []
    assert float(result["min_increment"]) == 0.0
    assert bool(result["monotone"]) is True
