"""Tests for aitcrg.compositional_regression."""

import math

from morie.fn import _array_core as np

from morie.fn.aitcrg import compositional_regression


def _ilr(y_row, V):
    """Compute ilr coordinates using contrast matrix V (D-by-(D-1))."""
    D = len(y_row)
    # geometric mean of the row
    lg = sum(math.log(v) for v in y_row) / D
    g = math.exp(lg)
    z = [y_row[i] / g for i in range(D)]
    # V is D-by-(D-1): column k is the ilr balance k
    Dv = len(V)
    q = len(V[0])
    out = []
    for k in range(q):
        s = 0.0
        for i in range(Dv):
            s += V[i][k] * math.log(z[i])
        out.append(s)
    return out


def test_aitcrg_basic():
    """Test basic functionality.

    Y_comp must be an N-by-D matrix of strictly positive compositions,
    and V must be a D-by-(D-1) contrast matrix. The previous version of
    this test passed 1-D arrays, which violates the documented shapes.
    """
    rng = np.random.default_rng(42)
    N, p, D = 100, 5, 4

    # Design matrix: N-by-p, used verbatim by the function.
    X = rng.normal(0.0, 1.0, (N, p))

    # Strictly positive compositions: N-by-D, all entries > 0.
    Y_comp = rng.uniform(0.1, 5.0, (N, D))

    # Contrast matrix V: D-by-(D-1) = 4-by-3 in this test.
    V = rng.normal(0.0, 1.0, (D, D - 1))

    result = compositional_regression(X, Y_comp, V)

    # The implementation returns a RichResult which behaves like a dict.
    assert isinstance(result, dict)

    # All advertised keys must be present.
    for key in ("beta", "fitted", "resid", "fitted_comp", "Y_ilr", "sse"):
        assert key in result

    # Scalar summary keys.
    for key in ("estimate", "N", "p", "D", "method"):
        assert key in result
    assert result["N"] == N
    assert result["p"] == p
    assert result["D"] == D

    # Shapes: beta is p-by-(D-1); fitted, resid, Y_ilr, fitted_comp are
    # N-by-(D-1) or N-by-D respectively.
    q = D - 1
    assert len(result["beta"]) == p
    for row in result["beta"]:
        assert len(row) == q

    assert len(result["fitted"]) == N
    for row in result["fitted"]:
        assert len(row) == q

    assert len(result["resid"]) == N
    for row in result["resid"]:
        assert len(row) == q

    assert len(result["Y_ilr"]) == N
    for row in result["Y_ilr"]:
        assert len(row) == q

    assert len(result["fitted_comp"]) == N
    for row in result["fitted_comp"]:
        assert len(row) == D
        for v in row:
            assert v > 0.0

    # estimate == beta[0][0]
    assert math.isclose(result["estimate"], result["beta"][0][0], rel_tol=1e-12)

    # sse equals the sum of squared residuals.
    sse_from_resid = 0.0
    for row in result["resid"]:
        for v in row:
            sse_from_resid += v * v
    assert math.isclose(result["sse"], sse_from_resid, rel_tol=1e-12)

    # The returned Y_ilr must agree with an independent ilr computation
    # using the supplied contrast matrix V (and Y_comp's row geometric means).
    V_list = V.tolist()
    Y_list = Y_comp.tolist()
    for i in range(N):
        expected_ilr = _ilr(Y_list[i], V_list)
        got_ilr = result["Y_ilr"][i]
        for k in range(q):
            assert math.isclose(got_ilr[k], expected_ilr[k], rel_tol=1e-10, abs_tol=1e-10)

    # The fitted ilr coordinates must equal X @ beta.
    X_list = X.tolist()
    for i in range(N):
        for k in range(q):
            s = 0.0
            for j in range(p):
                s += X_list[i][j] * result["beta"][j][k]
            assert math.isclose(result["fitted"][i][k], s, rel_tol=1e-10, abs_tol=1e-10)


def test_aitcrg_edge():
    """Test edge cases.

    The previous version of this test passed 1-D arrays of length 100,
    which is neither an N-by-p design matrix nor an N-by-D composition
    matrix nor a D-by-(D-1) contrast matrix. Pass 2-D shapes throughout.
    """
    rng = np.random.default_rng(42)
    N, p, D = 100, 5, 4

    X = rng.normal(0.0, 1.0, (N, p))
    Y_comp = rng.uniform(0.1, 5.0, (N, D))
    V = rng.normal(0.0, 1.0, (D, D - 1))

    result = compositional_regression(X, Y_comp, V)

    # Returned object supports dict-style access.
    assert isinstance(result, dict)
    assert "beta" in result
    assert "sse" in result
    assert result["N"] == N
    assert result["p"] == p
    assert result["D"] == D

    # SSE must be non-negative.
    assert result["sse"] >= 0.0
