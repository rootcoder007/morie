"""Test basis_pursuit (bpdn)."""

import pytest

from morie.fn import _array_core as np

from morie.fn._containers import DescriptiveResult
from morie.fn.bpdn import basis_pursuit, bpdn


class TestBpdn:
    def test_basic(self):
        rng = np.random.default_rng(42)
        D = rng.standard_normal((50, 100))
        x = D[:, 7] * 2.0 + D[:, 22] * 1.0
        result = basis_pursuit(D, x, lambda_=0.01)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "basis_pursuit"

    def test_sparsity(self):
        # x = 5 d_10, so the lasso optimum has the single support {10}
        # with c_10 = 5 - lambda / ||d_10||^2. ISTA converges at O(1/k),
        # so it is run to convergence rather than stopped at 300 steps,
        # where it had not yet left the dense start.
        rng = np.random.default_rng(1)
        D = rng.standard_normal((30, 60))
        x = D[:, 10] * 5.0
        lam = 0.5
        r = basis_pursuit(D, x, lambda_=lam, max_iter=20000, tol=1e-12)
        c = [float(v) for v in r.extra["coeffs"]]
        assert r.extra["n_nonzero"] == 1
        assert max(range(60), key=lambda j: abs(c[j])) == 10
        d10 = [float(D[i, 10]) for i in range(30)]
        norm2 = sum(v * v for v in d10)
        assert c[10] == pytest.approx(5.0 - lam / norm2, rel=1e-9)
        # lasso optimality: |d_j' r| <= lambda off the support, = on it
        resid = [float(x[i]) - sum(float(D[i, j]) * c[j] for j in range(60))
                 for i in range(30)]
        for j in range(60):
            g = sum(float(D[i, j]) * resid[i] for i in range(30))
            if j == 10:
                assert g == pytest.approx(lam, rel=1e-8)
            else:
                assert abs(g) <= lam + 1e-8

    def test_alias(self):
        assert bpdn is basis_pursuit
