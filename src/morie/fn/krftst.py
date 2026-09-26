"""Kenward-Roger adjusted F test for fixed effects under a parameterised covariance."""

from . import _array_core as np
from ._richresult import RichResult
from ._stats_core import f as _fdist

__all__ = ["kenward_roger_test"]


def _inv(A):
    ev = [abs(float(v)) for v in np.linalg.eigvalsh(A)]
    return np.linalg.inv(A) if min(ev) > 1e-10 else np.linalg.pinv(A)


def kenward_roger_test(X, y, Sigma, dSigma, L, l0=None, d2Sigma=None):
    r"""Kenward-Roger bias-adjusted Wald F test of :math:`H_0: L\beta = l_0`.

    With :math:`\Phi = (X'\Sigma^{-1}X)^{-1}` at the REML estimate
    :math:`\hat\theta` and :math:`\Sigma_i = \partial\Sigma/\partial\theta_i`,
    Kenward & Roger (1997), as written in Schabenberger & Gotway (2005,
    eqs 6.53-6.54, p. 343), adjust the variance of the EGLS estimator,

    .. math::

        \Phi_A = \Phi + 2\Phi\Big\{\sum_{i,j} W_{ij}\big(Q_{ij} - P_i\Phi P_j
        - \tfrac14 R_{ij}\big)\Big\}\Phi,

    :math:`P_i = -X'\Sigma^{-1}\Sigma_i\Sigma^{-1}X`,
    :math:`Q_{ij} = X'\Sigma^{-1}\Sigma_i\Sigma^{-1}\Sigma_j\Sigma^{-1}X`,
    :math:`R_{ij} = X'\Sigma^{-1}(\partial^2\Sigma/\partial\theta_i\partial\theta_j)\Sigma^{-1}X`,
    and :math:`W = \widehat{\mathrm{Var}}[\hat\theta]` twice the inverse of
    :math:`\mathrm{tr}(P\Sigma_iP\Sigma_j)`, the REML expected information
    (:math:`P = \Sigma^{-1} - \Sigma^{-1}X\Phi X'\Sigma^{-1}`). The test
    statistic is :math:`F^* = \lambda\,(L\hat\beta - l_0)'(L\Phi_AL')^{-1}
    (L\hat\beta - l_0)/q` on :math:`q = \mathrm{rank}(L)` and :math:`m`
    degrees of freedom, :math:`\lambda` and :math:`m` from the moment
    matching of Kenward & Roger (1997), as in pbkrtest.

    When :math:`\Sigma` is linear in :math:`\theta` (variance components)
    the :math:`R_{ij}` vanish and this is pbkrtest's KRmodcomp. For spatial
    covariance functions they do not; pass ``d2Sigma`` to include them,
    as (6.53) does.

    Parameters
    ----------
    X : array-like, (n, p)
    y : array-like, (n,)
    Sigma : array-like, (n, n)
        :math:`\Sigma(\hat\theta)`.
    dSigma : list of (n, n)
        :math:`\partial\Sigma/\partial\theta_i` at :math:`\hat\theta`.
    L : array-like, (q, p)
    l0 : array-like, (q,), optional
        Default zero.
    d2Sigma : dict, optional
        ``{(i, j): matrix}`` for the non-zero second derivatives,
        ``i <= j``.

    Returns
    -------
    RichResult
        ``beta``, ``Phi``, ``Phi_adjusted``, ``F`` (adjusted), ``ndf``,
        ``ddf`` (m), ``scaling`` (lambda), ``p_value``, ``F_unadjusted``,
        ``p_value_unadjusted`` (F on q and m with :math:`\Phi_A`, no
        scaling), ``W``.

    References
    ----------
    Kenward, M. G. & Roger, J. H. (1997). Small sample inference for fixed
    effects from restricted maximum likelihood. Biometrics 53, 983-997.
    Halekoh, U. & Hojsgaard, S. (2014). A Kenward-Roger approximation and
    parametric bootstrap methods for tests in linear mixed models -- the R
    package pbkrtest. Journal of Statistical Software 59(9).
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC, eqs (6.53)-(6.54), p. 343.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    S = np.asarray(Sigma, dtype=float)
    L = np.atleast_2d(np.asarray(L, dtype=float))
    G = [np.asarray(g, dtype=float) for g in dSigma]
    ng = len(G)
    Si = np.linalg.inv(S)
    TT = Si @ X
    Phi = np.linalg.inv(X.T @ TT)
    beta = Phi @ (TT.T @ y)
    H = [g @ Si for g in G]
    OX = [h @ X for h in H]
    P = [-(o.T @ TT) for o in OX]
    P = [0.5 * (m + m.T) for m in P]
    Q = {(i, j): OX[i].T @ Si @ OX[j] for i in range(ng) for j in range(ng)}
    IE2 = np.zeros((ng, ng))
    for i in range(ng):
        for j in range(ng):
            IE2[i, j] = (
                float(np.trace(H[i] @ H[j]))
                - 2.0 * float(np.sum(Phi * Q[(i, j)]))
                + float(np.trace(Phi @ P[i] @ Phi @ P[j]))
            )
    W = 2.0 * _inv(IE2)
    U = np.zeros(Phi.shape)
    for i in range(ng):
        for j in range(ng):
            term = Q[(i, j)] - P[i] @ Phi @ P[j]
            if d2Sigma is not None:
                key = (min(i, j), max(i, j))
                if key in d2Sigma:
                    term = term - 0.25 * (TT.T @ np.asarray(d2Sigma[key], dtype=float) @ TT)
            U = U + float(W[i, j]) * term
    U = 0.5 * (U + U.T)
    PhiA = Phi + 2.0 * (Phi @ U @ Phi)
    q = int(np.linalg.matrix_rank(L))
    Theta = L.T @ np.linalg.solve(L @ Phi @ L.T, L)
    A1 = A2 = 0.0
    for i in range(ng):
        ui = Theta @ Phi @ P[i] @ Phi
        for j in range(ng):
            uj = Theta @ Phi @ P[j] @ Phi
            A1 += float(W[i, j]) * float(np.trace(ui)) * float(np.trace(uj))
            A2 += float(W[i, j]) * float(np.trace(ui @ uj))
    B = (A1 + 6.0 * A2) / (2.0 * q)
    g = ((q + 1) * A1 - (q + 4) * A2) / ((q + 2) * A2)
    den = 3.0 * q + 2.0 * (1.0 - g)
    c1, c2, c3 = g / den, (q - g) / den, (q + 2 - g) / den
    V0 = 1.0 + c1 * B
    V0 = 0.0 if abs(V0) < 1e-10 else V0
    V1, V2 = 1.0 - c2 * B, 1.0 - c3 * B
    rho = (1.0 / q) * ((1.0 - A2 / q) / V1) ** 2 * V0 / V2
    m = 4.0 + (q + 2.0) / (q * rho - 1.0)
    lam = 1.0 if abs(m - 2.0) < 0.01 else m * (1.0 - A2 / q) / (m - 2.0)
    l0 = np.zeros(q) if l0 is None else np.asarray(l0, dtype=float).ravel()
    d = L @ beta - l0
    wald = float(d @ np.linalg.solve(L @ PhiA @ L.T, d))
    Fu = wald / q
    F = lam * Fu
    return RichResult(
        title="Kenward-Roger F test (eqs 6.53-6.54)",
        summary_lines=[("F", F), ("ndf", q), ("ddf", m), ("p-value", float(_fdist.sf(F, q, m)))],
        payload={
            "beta": [float(v) for v in beta],
            "Phi": Phi,
            "Phi_adjusted": PhiA,
            "F": F,
            "ndf": q,
            "ddf": m,
            "scaling": lam,
            "p_value": float(_fdist.sf(F, q, m)),
            "F_unadjusted": Fu,
            "p_value_unadjusted": float(_fdist.sf(Fu, q, m)),
            "W": W,
            "A1": A1,
            "A2": A2,
        },
    )


def cheatsheet():
    return "krftst: Kenward-Roger F with the bias-adjusted Phi_A, lambda and m (6.53-6.54)"
