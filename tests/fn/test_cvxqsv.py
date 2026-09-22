"""Tests for cvxqsv.boyd_qcqp_relaxation."""

from morie.fn import _array_core as np

from morie.fn.cvxqsv import boyd_qcqp_relaxation


def test_cvxqsv_basic():
    """Test basic functionality on the documented convex QCQP example.

    Minimise ``|x|^2/2 - 2*x1`` over the unit disc. The relaxation
    should give the true optimum, come back rank one, and reproduce
    ``x = (1, 0)``.
    """
    P0 = np.eye(2)
    q0 = [-2.0, 0.0]
    P = [np.eye(2)]
    q = [[0.0, 0.0]]
    r = [-0.5]

    result = boyd_qcqp_relaxation(P0, q0, P=P, q=q, r=r)

    assert isinstance(result, dict)

    # Every documented return key must be present.
    for key in ("X", "x", "lower_bound", "rank", "eigenvalues",
                "tight", "residual", "gap_bound", "converged"):
        assert key in result

    # The convex case: relaxation is tight, rank is one, x is the
    # known optimum of the original problem, and the lower bound
    # equals the true value -3/2.
    assert bool(result["tight"]) is True
    assert int(result["rank"]) == 1
    assert round(float(result["lower_bound"]), 4) == -1.5
    assert round(float(result["x"][0]), 4) == 1.0
    assert round(float(result["x"][1]), 4) == 0.0

    # Residual ||X - x x^T|| must be tiny when the relaxation is tight.
    X = result["X"]
    x = result["x"]
    expected_residual = float(np.linalg.norm(X - np.outer(x, x)))
    assert float(result["residual"]) < 1e-5

    # Lower bound, recomputed independently from the formula on
    # (X, x): 1/2 <P0, X> + q0 . x, evaluated at the returned point.
    lb_recomputed = 0.5 * float(np.dot(P0.ravel(), X.ravel())) + float(np.dot(q0, x))
    assert round(lb_recomputed, 3) == round(float(result["lower_bound"]), 3)


def test_cvxqsv_edge():
    """Test that the documented nonconvex (frustrated triangle) example
    returns a rank-2 X and a lower bound strictly below the integral
    optimum -1, with eigenvalues {0, 3/2, 3/2}."""
    J = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    q0 = np.zeros(3)
    P = [2.0 * np.outer(e, e) for e in np.eye(3)]
    q = [np.zeros(3) for _ in range(3)]
    r = [-1.0, -1.0, -1.0]

    result = boyd_qcqp_relaxation(J, q0, P=P, q=q, r=r)

    assert isinstance(result, dict)
    for key in ("X", "x", "lower_bound", "rank", "eigenvalues",
                "tight", "residual", "gap_bound", "converged"):
        assert key in result

    # Lower bound is -3/2, strictly below the true minimum -1.
    assert round(float(result["lower_bound"]), 3) == -1.5
    assert bool(result["lower_bound"] < -1.0) is True

    # The relaxation lost something: rank is 2, not 1.
    assert int(result["rank"]) == 2
    assert bool(result["tight"]) is False

    # Eigenvalues of X match the 120-degree construction: {0, 3/2, 3/2}.
    eigvals = sorted(float(v) for v in np.linalg.eigvalsh(result["X"]))
    assert round(eigvals[0], 4) == 0.0
    assert round(eigvals[1], 4) == 1.5
    assert round(eigvals[2], 4) == 1.5
