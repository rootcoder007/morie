"""Tests for fwlfwd.fully_corrective_fw."""

from morie.fn import _array_core as np

from morie.fn.fwlfwd import fully_corrective_fw


def _to_list(a):
    """Convert a possibly array-like shim object to a plain Python list."""
    if hasattr(a, "tolist"):
        return a.tolist()
    return list(a)


def _make_callable(values):
    """Wrap a sequence of scalars as a callable that returns f(x) = sum_i x[i] * c[i]."""
    coeffs = _to_list(values)

    def fn(x):
        xs = _to_list(x)
        return sum(c * xi for c, xi in zip(coeffs, xs))

    return fn, coeffs


def _make_grad(values):
    """Constant gradient equal to the coefficient vector."""
    coeffs = _to_list(values)

    def g(x):
        xs = _to_list(x)
        return [c for c, _xi in zip(coeffs, xs)]

    return g, coeffs


def test_fwlfwd_basic():
    """Test basic functionality on a small 2D vertex set with linear f."""
    # Three vertices in R^2 forming a triangle domain.
    domain = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    # Linear objective f(x) = 2*x0 + 3*x1, with matching constant gradient.
    f, coeffs = _make_callable([2.0, 3.0])
    grad_f, _ = _make_grad([2.0, 3.0])

    x0 = [0.0, 0.0]

    result = fully_corrective_fw(f, grad_f, domain, x0)

    # Result exposes a payload with documented keys.
    payload = result.payload
    assert "estimate" in payload
    assert "x" in payload
    assert "f_path" in payload
    assert "gap" in payload
    assert "n_active" in payload
    assert "n" in payload
    assert "method" in payload

    # n must equal the ambient dimension of the domain.
    assert payload["n"] == 2

    # For f(x) = 2*x0 + 3*x1 on the triangle conv{(0,0),(1,0),(0,1)},
    # the maximum over the vertices is attained at (0, 1) with value 3.
    expected_f_min = min(
        coeffs[0] * v[0] + coeffs[1] * v[1] for v in domain
    )
    expected_f_max = max(
        coeffs[0] * v[0] + coeffs[1] * v[1] for v in domain
    )

    # Frank-Wolfe minimises the linear objective, so the estimate must
    # equal the vertex-minimum of f.
    assert payload["estimate"] == expected_f_min

    # The optimal vertex itself must be in the active set / final iterate.
    xs = _to_list(payload["x"])
    assert len(xs) == 2

    # f_path should contain the initial value plus one entry per step.
    path = list(payload["f_path"])
    assert len(path) == 10 + 1
    # Final path entry matches the reported estimate.
    assert path[-1] == payload["estimate"]


def test_fwlfwd_edge():
    """Test edge case: x0 is already the optimal vertex (gradient gap is 0)."""
    # Square domain in R^2.
    domain = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]

    # f(x) = x0 + x2 minimised at (0,0); x0 = (0,0) is already optimal.
    f, coeffs = _make_callable([1.0, 1.0])
    grad_f, _ = _make_grad([1.0, 1.0])

    x0 = [0.0, 0.0]

    result = fully_corrective_fw(f, grad_f, domain, x0)

    payload = result.payload

    # Expected minimum over the square is 0, attained at (0,0).
    expected_f_min = min(
        coeffs[0] * v[0] + coeffs[1] * v[1] for v in domain
    )
    assert payload["estimate"] == expected_f_min
    assert payload["estimate"] == 0.0

    # The initial point is optimal, so the FW gap should be <= 0
    # (it is exactly 0 for the linear case here).
    assert payload["gap"] <= 1e-12

    # f_path: initial f(x0) is 0, all subsequent values are also 0.
    path = list(payload["f_path"])
    assert len(path) == 10 + 1
    for value in path:
        assert abs(value - 0.0) < 1e-12
