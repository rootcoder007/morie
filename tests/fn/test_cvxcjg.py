"""Tests for cvxcjg.boyd_conjugate."""

from morie.fn import _array_core as np

from morie.fn.cvxcjg import boyd_conjugate


def test_cvxcjg_basic():
    """Test basic functionality for the self-conjugate quadratic f(x) = x^2/2."""
    g = np.linspace(-10.0, 10.0, 20001)
    result = boyd_conjugate(lambda x: 0.5 * x ** 2, [1.0, 2.0, 3.0], x_grid=g)
    # The function returns a RichResult; assert key payload fields exist.
    assert hasattr(result, "payload")
    payload = result.payload
    assert "value" in payload
    assert "argmax" in payload
    assert "unbounded" in payload
    assert "supporting_intercept" in payload
    # For f(x) = x^2/2 the conjugate is f*(y) = y^2/2.
    y_test = [1.0, 2.0, 3.0]
    expected_values = [0.5 * yy * yy for yy in y_test]
    for got, exp in zip(payload["value"], expected_values):
        assert abs(float(got) - exp) < 1e-2
    # The maximiser satisfies f'(x) = y, i.e. x = y.
    for got, yy in zip(payload["argmax"], y_test):
        assert abs(float(got) - yy) < 1e-2


def test_cvxcjg_edge():
    """Test edge cases: |x| conjugate is 0 on [-1,1] and unbounded outside."""
    g = np.linspace(-10.0, 10.0, 20001)
    # Inside the dual-norm ball the conjugate value is 0.
    inside = boyd_conjugate(abs, 0.5, x_grid=g).payload
    assert isinstance(inside, dict)
    assert abs(inside["value"]) < 1e-6
    # Outside the dual-norm ball the conjugate is unbounded.
    outside = boyd_conjugate(abs, 2.0, x_grid=g).payload
    assert "value" in outside
    assert bool(outside["unbounded"]) is True
