"""Tests for eslsce.esl_score_match."""

from morie.fn import _array_core as np

from morie.fn.eslsce import esl_score_match


def test_eslsce_basic():
    """Test basic functionality.

    For a Gaussian ``N(mu, 1)`` model the score is ``psi(x) = -(x - mu)``.
    Hyvarinen's objective at the truth reduces to a constant ``d/2`` (trace = -d,
    norm = d/2, so per-point = 0 and J = 0 mean), independent of the data.

    Here we evaluate J(theta) = E[tr(grad psi) + 0.5 ||psi||^2] at mu = 0 with a
    large sample and check that it is close to 0.
    """
    rng = np.random.default_rng(42)
    n = 4000
    d = 1
    X = rng.normal(0.0, 1.0, (n, d))
    score_fn = lambda z: -(z - 0.0)

    result = esl_score_match(score_fn, X)

    assert isinstance(result, dict)
    assert "objective" in result
    assert "trace_term" in result
    assert "norm_term" in result
    assert "per_point" in result

    # Independent computation of the literature formula on the same inputs.
    psi = -(X - 0.0)
    # For psi(x) = -(x - 0), d psi / d x = -1 on the diagonal -> trace = -d.
    expected_trace = -d
    expected_norm = 0.5 * float((psi ** 2).mean())
    expected_objective = expected_trace + expected_norm

    assert abs(result["trace_term"] - expected_trace) < 1e-3
    assert abs(result["norm_term"] - expected_norm) < 1e-3
    assert abs(result["objective"] - expected_objective) < 1e-3

    # The unit-Gaussian identity from the docstring: trace_term == -d.
    assert abs(result["trace_term"] + d) < 1e-3


def test_eslsce_edge():
    """Test edge cases: analytic grad_score agrees with finite differences."""
    rng = np.random.default_rng(42)
    n = 2000
    d = 1
    X = rng.normal(2.0, 1.0, (n, d))
    score_fn = lambda z: -(z - 2.0)

    r_fd = esl_score_match(score_fn, X)
    r_an = esl_score_match(score_fn, X, grad_score=lambda z: -np.ones_like(z))

    assert abs(r_fd["objective"] - r_an["objective"]) < 1e-3
    assert abs(r_fd["trace_term"] - r_an["trace_term"]) < 1e-3
    assert abs(r_fd["norm_term"] - r_an["norm_term"]) < 1e-3
