"""Tests for drbnk.dr_bandit_did."""

from morie.fn import _array_core as np

from morie.fn.drbnk import dr_bandit_did


def test_drbnk_basic():
    """Test basic functionality with valid binary D_t and in-(0,1) pi_t."""
    n = 100
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    rng_x = np.random.default_rng(42)
    rng_p = np.random.default_rng(41)

    y = list(rng_y.normal(0.0, 1.0, n))

    # Build a binary D_t with a clear, known treated share (50/50).
    D_t = [1.0] * (n // 2) + [0.0] * (n - n // 2)
    rng_d.shuffle(D_t) if hasattr(rng_d, "shuffle") else None
    # Manual shuffle via python RNG to keep the shim happy:
    import random as _pyr
    _pyr.seed(42)
    _pyr.shuffle(D_t)

    # Covariate matrix (n, 5) still required by the regression code path.
    X = [list(rng_x.normal(0.0, 1.0, 5)) for _ in range(n)]

    # Known assignment probabilities strictly inside (0,1).
    pi_t = [0.2 + 0.6 * float(v) for v in rng_p.uniform(0.0, 1.0, n)]

    result = dr_bandit_did(y, D_t, X, pi_t)

    # Documented return keys.
    assert isinstance(result, dict)
    for key in ("estimate", "se", "aipw_unweighted",
                "h", "sum_h2_over_e", "n_treat", "n"):
        assert key in result, f"missing key: {key}"

    assert result["n"] == n
    assert result["n_treat"] == sum(D_t)

    # Independent recomputation of the adaptive weights h (eq. 12 of
    # Hadad et al. 2021, arXiv:1911.02768), using the same pi_t.
    acc = 0.0
    h_indep = []
    for i in range(n):
        lam = 1.0 / float(n - i)
        q = (1.0 - acc) * lam
        if q < 0.0:
            q = 0.0
        acc += q
        h_indep.append((q * pi_t[i]) ** 0.5)

    # Independent OLS via normal equations (closed-form), bypassing
    # whatever lstsq the shim uses.
    def _lstsq(Z, yy):
        # Z^T Z and Z^T y
        p = len(Z[0])
        ZtZ = [[0.0] * p for _ in range(p)]
        Zty = [0.0] * p
        for i in range(len(Z)):
            Zi = Z[i]
            yi = yy[i]
            for a in range(p):
                Zty[a] += Zi[a] * yi
                for b in range(p):
                    ZtZ[a][b] += Zi[a] * Zi[b]
        # solve via Gaussian elimination
        M = [row[:] + [Zty[i]] for i, row in enumerate(ZtZ)]
        for k in range(p):
            piv = M[k][k]
            for j in range(k, p + 1):
                M[k][j] /= piv
            for i in range(p):
                if i != k:
                    f = M[i][k]
                    for j in range(k, p + 1):
                        M[i][j] -= f * M[k][j]
        return [M[i][p] for i in range(p)]

    # Build the design matrix Z that the function uses (intercept + X).
    Z = [[1.0] + X[i] for i in range(n)]
    i1 = [i for i in range(n) if D_t[i] >= 0.5]
    i0 = [i for i in range(n) if D_t[i] < 0.5]
    b1 = _lstsq([Z[i] for i in i1], [y[i] for i in i1])
    b0 = _lstsq([Z[i] for i in i0], [y[i] for i in i0])

    def _mv(Zi, b):
        s = 0.0
        for kk, bk in enumerate(b):
            s += Zi[kk] * bk
        return s

    m1 = [_mv(Z[i], b1) for i in range(n)]
    m0 = [_mv(Z[i], b0) for i in range(n)]

    psi = []
    for i in range(n):
        psi.append(
            m1[i] - m0[i]
            + D_t[i] * (y[i] - m1[i]) / pi_t[i]
            - (1.0 - D_t[i]) * (y[i] - m0[i]) / (1.0 - pi_t[i])
        )

    aipw_unweighted_indep = sum(psi) / float(n)

    sh = sum(h_indep)
    est_indep = sum(h_indep[i] * psi[i] for i in range(n)) / sh

    v = sum((h_indep[i] * (psi[i] - est_indep)) ** 2 for i in range(n))
    se_indep = (v ** 0.5) / sh

    assert abs(result["sum_h2_over_e"] - acc) < 1e-9
    assert abs(result["aipw_unweighted"] - aipw_unweighted_indep) < 1e-9
    assert abs(result["estimate"] - est_indep) < 1e-9
    assert abs(result["se"] - se_indep) < 1e-9
    assert result["se"] >= 0.0


def test_drbnk_edge():
    """Test edge cases: still-valid inputs plus both documented result keys."""
    n = 100
    rng_y = np.random.default_rng(43)
    rng_d = np.random.default_rng(42)
    rng_x = np.random.default_rng(42)
    rng_p = np.random.default_rng(41)

    y = list(rng_y.normal(0.0, 1.0, n))

    # 50/50 binary assignment, reshuffled.
    D_t = [1.0] * (n // 2) + [0.0] * (n - n // 2)
    import random as _pyr
    _pyr.seed(7)
    _pyr.shuffle(D_t)

    X = [list(rng_x.normal(0.0, 1.0, 5)) for _ in range(n)]

    # Probabilities strictly inside (0,1).
    pi_t = [0.2 + 0.6 * float(v) for v in rng_p.uniform(0.0, 1.0, n)]

    result = dr_bandit_did(y, D_t, X, pi_t)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "se" in result
    assert "h" in result
    assert "sum_h2_over_e" in result
    assert "n_treat" in result
    assert "n" in result
    assert result["n"] == n
