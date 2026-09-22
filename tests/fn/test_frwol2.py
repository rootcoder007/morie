"""Tests for frwol2.frank_wolfe."""

from morie.fn import _array_core as np

from morie.fn.frwol2 import frank_wolfe


def _make_callable(values):
    """Wrap an array into a zero-d callable returning a Python float."""
    arr = [float(v) for v in values]

    def fn(_x=None):
        # return the first element as a scalar float
        return float(arr[0])

    return fn


def _make_grad(values):
    """Wrap an array into a callable returning a constant gradient vector."""
    arr = [float(v) for v in values]

    def gfn(_x):
        return list(arr)

    return gfn


def test_frwol2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)

    # Generate a domain: a list of vertices, each vertex is a list of d numbers.
    # Here d = 5 and we have 10 vertices.
    d = 5
    n_vertices = 10
    domain = [rng.normal(0.0, 1.0, d).tolist() for _ in range(n_vertices)]

    # x0 must have dimension d.
    x0 = rng.normal(0.0, 1.0, d).tolist()

    # f and grad_f must be callables; the function does not care about their
    # values for the test of the algorithm's wiring.
    f_vals = rng.normal(0.0, 1.0, 1).tolist()
    grad_vals = rng.normal(0.0, 1.0, d).tolist()
    f = _make_callable(f_vals)
    grad_f = _make_grad(grad_vals)

    steps = 5  # an int, not an array

    result = frank_wolfe(f, grad_f, domain, x0, steps)

    # The function returns a RichResult (dict-like), with documented payload keys.
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "x" in result
    assert "f_path" in result
    assert "gap" in result
    assert "steps" in result
    assert "n" in result
    assert "method" in result

    # Numeric checks computed independently from the documented formula.
    assert result["steps"] == steps
    assert result["n"] == d
    assert len(result["f_path"]) == steps + 1

    # Independent recomputation of the frank-wolfe update for t = 0.
    t = 0
    gamma = 2.0 / (t + 2.0)  # 1.0
    i0 = min(range(len(domain)), key=lambda j: sum(grad_vals[k] * domain[j][k]
                                                   for k in range(d)))
    expected_x = [(1.0 - gamma) * x0[k] + gamma * domain[i0][k] for k in range(d)]
    # x at iteration 0 must match the independently computed expected_x.
    assert len(result["x"]) == d
    for k in range(d):
        assert abs(result["x"][k] - expected_x[k]) < 1e-12


def test_frwol2_edge():
    """Test edge cases: minimal valid input with steps = 1."""
    rng = np.random.default_rng(42)

    d = 3
    domain = [rng.normal(0.0, 1.0, d).tolist() for _ in range(4)]
    x0 = rng.normal(0.0, 1.0, d).tolist()

    f = _make_callable(rng.normal(0.0, 1.0, 1).tolist())
    grad_f = _make_grad(rng.normal(0.0, 1.0, d).tolist())

    steps = 1
    result = frank_wolfe(f, grad_f, domain, x0, steps)

    assert isinstance(result, dict)
    assert result["steps"] == 1
    assert result["n"] == d
    assert len(result["f_path"]) == 2  # initial + one update

    # The first entry of f_path is f(x0).
    assert result["f_path"][0] == float(f(x0))
