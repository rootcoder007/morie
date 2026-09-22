"""Tests for cmaopt.cma_es."""

import math

from morie.fn import _array_core as np

from morie.fn.cmaopt import cma_es


def test_cmaopt_basic():
    """Test basic functionality on the spherical function f(x)=sum(x**2)."""
    N = 5
    iters = 4
    lam = 6  # population size

    def f(x):
        return float(sum(v * v for v in x))

    x0 = list(np.random.default_rng(0).normal(0.0, 1.0, N))
    Z = np.random.default_rng(1).normal(0.0, 1.0, (iters * lam, N))

    sigma = 1.0
    result = cma_es(f, x0, sigma, Z, lam, iters)

    # --- return-type / key contract ---
    # RichResult exposes payload via attribute access; confirm it is a CMA-ES result
    # by checking the documented keys.
    assert hasattr(result, "estimate")
    assert hasattr(result, "fbest")
    assert hasattr(result, "xbest")
    assert hasattr(result, "xmean")
    assert hasattr(result, "sigma")
    assert hasattr(result, "C")
    assert hasattr(result, "evals")
    assert hasattr(result, "generations")
    assert hasattr(result, "n")
    assert hasattr(result, "method")

    # --- value sanity ---
    assert int(result.generations) == iters
    assert int(result.n) == N
    assert int(result.evals) == iters * lam
    assert float(result.sigma) > 0.0

    # f(x)=sum(x**2) is non-negative; CMA-ES minimises it.
    assert float(result.fbest) >= 0.0
    assert float(result.estimate) == float(result.fbest)

    # xbest must be a length-N vector; reconstruct the objective from it
    # independently and verify it matches fbest to within FP tolerance.
    xb = [float(v) for v in np.asarray(result.xbest).ravel().tolist()]
    assert len(xb) == N
    expected_fbest = sum(v * v for v in xb)
    assert math.isclose(float(result.fbest), expected_fbest, rel_tol=1e-12, abs_tol=1e-12)

    # xmean must also be a length-N vector and finite.
    xm = [float(v) for v in np.asarray(result.xmean).ravel().tolist()]
    assert len(xm) == N
    for v in xm:
        assert math.isfinite(v)

    # C must be an N x N symmetric positive-ish matrix (entries finite).
    C = np.asarray(result.C)
    assert C.shape == (N, N)
    for r in range(N):
        for c in range(N):
            assert math.isfinite(float(C[r, c]))
        # symmetry
        for c in range(N):
            assert math.isclose(float(C[r, c]), float(C[c, r]),
                                rel_tol=1e-12, abs_tol=1e-12)


def test_cmaopt_edge():
    """Test edge cases: 1-D problem with the default lam heuristic."""
    N = 3
    iters = 2
    # lam must be a positive int; let cma_es default lam = 4 + floor(3 ln N)
    expected_lam_default = 4 + int(3.0 * math.log(float(N)))
    lam = expected_lam_default

    def f(x):
        return float(sum(v * v for v in x))

    x0 = [0.0] * N
    Z = np.random.default_rng(2).normal(0.0, 1.0, (iters * lam, N))

    sigma = 0.5
    # pass lam explicitly to match the Z-array shape; function still defaults
    # the same way internally.
    result = cma_es(f, x0, sigma, Z, lam, iters)

    assert hasattr(result, "fbest")
    assert hasattr(result, "n")
    assert int(result.n) == N
    assert int(result.generations) == iters
    assert int(result.evals) == iters * lam
    assert float(result.fbest) >= 0.0
    assert float(result.sigma) > 0.0
