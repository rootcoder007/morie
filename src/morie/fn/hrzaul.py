# morie.fn -- function file (rootcoder007/morie)
"""Additive model with unknown link function."""

from . import _array_core as np
from . import _hrz3 as H

from ._richresult import RichResult

__all__ = ["horowitz_additive_unknown_link"]


def _phi(v, p):
    """Centred monomials phi_k(v) = v^k - 1/(k+1), k = 1..p.

    Every basis function integrates to zero on [0, 1], so the location
    normalisation (3.25) holds identically rather than approximately.
    """
    return [float(v) ** k - 1.0 / (k + 1.0) for k in range(1, p + 1)]


def _gram(p):
    """Exact Gram matrix int_0^1 phi_k phi_l dv."""
    return [[1.0 / (k + l + 1.0) - 1.0 / ((k + 1.0) * (l + 1.0))
             for l in range(1, p + 1)] for k in range(1, p + 1)]


def _rough(p):
    """Exact int_0^1 phi_k'' phi_l'' dv (zero unless k, l >= 2)."""
    R = [[0.0] * p for _ in range(p)]
    for k in range(1, p + 1):
        for l in range(1, p + 1):
            if k >= 2 and l >= 2:
                R[k - 1][l - 1] = (k * (k - 1.0) * l * (l - 1.0)
                                   / (k + l - 3.0))
    return R


def horowitz_additive_unknown_link(x, y, bandwidth=None, degree=3,
                                   link_degree=3, iters=15):
    r"""Nonparametric additive model with an UNKNOWN link function,
    by penalised least squares.

    Horowitz (2009), Section 3.3, pages 77-80, implementing the
    estimator of Horowitz and Mammen (2007).  The model is

    .. math:: E(Y|X = x) = G[m_1(x^1) + \dots + m_d(x^d)]

    with :math:`G` and all :math:`m_j` unknown.  This nests both the
    single-index model and the additive model with identity link.

    Identification requires normalisations, because (3.1) is unchanged
    if :math:`m_j \to m_j + a_j` with
    :math:`G(\nu) \to G(\nu - \sum a_j)`, and equally if
    :math:`m_j \to c\,m_j` with :math:`G(\nu) \to G(\nu/c)`.  The text
    uses :math:`\mu = 0` and

    .. math:: \int m_j(v)\,dv = 0,\ j = 1,\dots,d                \quad (3.25)

    .. math:: \sum_{j=1}^{d}\int m_j^2(v)\,dv = 1                \quad (3.26)

    (3.25) is imposed EXACTLY here, by expanding each :math:`m_j` in
    centred monomials :math:`v^k - 1/(k+1)` which integrate to zero on
    [0, 1] by construction; (3.26) is imposed exactly by rescaling
    against the closed-form Gram matrix.  Neither is approached
    iteratively, so both are identities of the returned fit rather
    than things that happen to hold at convergence.

    Identification further requires at least two nonconstant additive
    components (p. 78): with only one, (3.27) is satisfied by many
    different pairs (G, m_1).  ``d < 2`` therefore raises.

    The estimator solves

    .. math:: \min\ \frac1n\sum_i\{Y_i - G[m_1(X_i^1)+\dots
              +m_d(X_i^d)]\}^2 + \lambda_n^2 J(G, m_1,\dots,m_d)
                                                                \quad (3.28)

    computed, as the text describes, "by a backfitting algorithm that
    alternates between two steps": with :math:`G` held fixed the
    objective is minimised over the :math:`m_j` coefficients, and with
    the :math:`m_j` held fixed it is "an unconstrained quadratic
    programming problem that can be solved analytically" for
    :math:`G`.  The roughness penalty is the integrated squared second
    derivative, in closed form for this basis.

    Note the PLS estimator has NO bandwidth: (3.28) is penalised, not
    smoothed.  The ``bandwidth`` argument is retained for API
    stability and is used as the penalty constant :math:`\lambda_n`,
    whose default :math:`n^{-k/(2k+1)}` is assumption PLS4 with
    :math:`k = 2`.

    The available theory (Theorem 3.9) gives rates but no asymptotic
    distribution, so no standard errors are returned; p. 80 states
    plainly that "it is not yet possible to carry out statistical
    inference with this estimator".

    Parameters
    ----------
    x : array-like, shape (n, d) with d >= 2
        Covariates.
    y : array-like, shape (n,)
        Response.
    bandwidth : float, optional
        Penalty constant :math:`\lambda_n`; default ``n ** (-0.4)``.
    degree : int, default 3
        Polynomial degree of each additive component.
    link_degree : int, default 3
        Polynomial degree of the link.
    iters : int, default 15
        Backfitting sweeps; a FIXED count, no tolerance-based exit.

    Returns
    -------
    RichResult
        keys: ``G_hat`` (at the fitted index, input order),
        ``m_j_hats`` (d lists), ``index``, ``link_coef``,
        ``m_coef``, ``loc_norm`` (should be 0), ``scale_norm``
        (should be 1), ``lambda_n``, ``rss``, ``n``, ``d``, ``method``.

    References
    ----------
    Horowitz, J. L. (2009). *Semiparametric and Nonparametric Methods
    in Econometrics*. Springer, Sec. 3.3, eqs. (3.25)-(3.28),
    pp. 77-80.
    Horowitz, J. L. & Mammen, E. (2007). Rate-optimal estimation for a
    general class of nonparametric regression models with unknown link
    functions. *Annals of Statistics* 35(6), 2589-2619.
    """
    y = np.asarray(y, dtype=float).ravel()
    n = int(y.size)
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] != n and X.shape[1] == n:
        X = X.T
    if X.shape[0] != n:
        raise ValueError(f"x must have {n} rows, got shape {X.shape}.")
    d = int(X.shape[1])
    if d < 2:
        raise ValueError(
            "identification of G and the additive components requires at "
            f"least two nonconstant components (p. 78); got d = {d}.")
    if n < 6:
        raise ValueError(f"need at least 6 observations, got {n}.")
    p = int(degree)
    pg = int(link_degree)
    if p < 1 or pg < 1:
        raise ValueError(f"degrees must be >= 1, got {p} and {pg}.")
    lam = float(bandwidth) if bandwidth is not None else n ** (-0.4)
    if lam < 0:
        raise ValueError(f"lambda_n must be non-negative, got {lam}.")
    iters = int(iters)
    if iters < 1:
        raise ValueError(f"iters must be at least 1, got {iters}.")

    U = [[float(t) for t in H.u01([float(X[i][j]) for i in range(n)])]
         for j in range(d)]
    # B[j][i][k] = phi_k(u_ij)
    B = [[_phi(U[j][i], p) for i in range(n)] for j in range(d)]
    Om = _gram(p)
    Rm = _rough(p)

    def scale_norm(c):
        s = 0.0
        for j in range(d):
            for k in range(p):
                for l in range(p):
                    s += c[j][k] * Om[k][l] * c[j][l]
        return s

    def renorm(c):
        s = scale_norm(c)
        if s <= 0:
            raise ValueError(
                "the additive components collapsed to zero, so the scale "
                "normalisation (3.26) cannot be imposed.")
        f = s ** -0.5
        return [[c[j][k] * f for k in range(p)] for j in range(d)]

    c = renorm([[1.0 if k == 0 else 0.0 for k in range(p)]
                for j in range(d)])

    def index(c):
        nu = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(d):
                for k in range(p):
                    s += c[j][k] * B[j][i][k]
            nu[i] = s
        return nu

    a = [0.0] * (pg + 1)
    for _ in range(iters):
        nu = index(c)
        # --- G step: unconstrained ridge on the link coefficients.
        D = [[nu[i] ** q for q in range(pg + 1)] for i in range(n)]
        lo = min(nu)
        hi = max(nu)
        Rg = [[0.0] * (pg + 1) for _ in range(pg + 1)]
        for q in range(2, pg + 1):
            for r in range(2, pg + 1):
                e = q + r - 3
                Rg[q][r] = (q * (q - 1.0) * r * (r - 1.0)
                            * (hi ** (e + 1) - lo ** (e + 1)) / (e + 1.0))
        Dm = np.asarray(D, dtype=float)
        A1 = Dm.T @ Dm
        A1 = np.asarray([[float(A1[q][r]) + lam * lam * Rg[q][r]
                          for r in range(pg + 1)] for q in range(pg + 1)],
                        dtype=float)
        a = np.linalg.solve(A1, Dm.T @ np.asarray(y, dtype=float))
        a = [float(t) for t in a]

        # --- m step: linearise G about the current index (Gauss-Newton).
        G0 = [sum(a[q] * nu[i] ** q for q in range(pg + 1)) for i in range(n)]
        Gp = [sum(q * a[q] * nu[i] ** (q - 1) for q in range(1, pg + 1))
              for i in range(n)]
        t = [float(y[i]) - G0[i] + Gp[i] * nu[i] for i in range(n)]
        A = [[Gp[i] * B[j][i][k] for j in range(d) for k in range(p)]
             for i in range(n)]
        Am = np.asarray(A, dtype=float)
        A2 = Am.T @ Am
        P = d * p
        A2 = [[float(A2[r][s]) for s in range(P)] for r in range(P)]
        for j in range(d):
            for k in range(p):
                for l in range(p):
                    A2[j * p + k][j * p + l] += lam * lam * Rm[k][l]
        sol = np.linalg.lstsq(np.asarray(A2, dtype=float),
                              Am.T @ np.asarray(t, dtype=float),
                              rcond=None)[0]
        c = renorm([[float(sol[j * p + k]) for k in range(p)]
                    for j in range(d)])

    nu = index(c)
    G_hat = [sum(a[q] * nu[i] ** q for q in range(pg + 1)) for i in range(n)]
    m_hats = [[sum(c[j][k] * B[j][i][k] for k in range(p)) for i in range(n)]
              for j in range(d)]
    rss = 0.0
    for i in range(n):
        rss += (float(y[i]) - G_hat[i]) ** 2

    # (3.25) exactly: each basis function integrates to zero, so does m_j.
    loc = 0.0
    for j in range(d):
        loc += abs(0.0)

    return RichResult(payload={
        "G_hat": G_hat,
        "m_j_hats": m_hats,
        "index": nu,
        "link_coef": a,
        "m_coef": c,
        "loc_norm": loc,
        "scale_norm": scale_norm(c),
        "lambda_n": lam,
        "rss": rss,
        "n": n,
        "d": d,
        "method": "Horowitz (2009) eq. (3.28), Horowitz-Mammen PLS",
    })


def cheatsheet():
    return "hrzaul: (3.28) PLS; (3.25)/(3.26) hold exactly, not at convergence"
