# morie.fn -- function file (rootcoder007/morie)
"""BLUE and BLUP from Henderson's mixed model equations."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["blue_gls"]


def blue_gls(y, X, V=None, Z=None, Sigma=None, R=None, K=None):
    r"""Generalised least squares fixed effects, with BLUPs when asked.

    For the linear mixed model :math:`Y = X\beta + Zu + \varepsilon`
    with :math:`u \sim N(0, \Sigma)` and
    :math:`\varepsilon \sim N(0, R)`, the marginal variance is
    :math:`V = Z\Sigma Z^{T} + R` and the best linear unbiased
    estimator of :math:`\beta` is Aitken's

    .. math:: \hat\beta = (X^{T}V^{-1}X)^{-1}X^{T}V^{-1}y .

    Given ``Z``, ``Sigma`` and ``R`` this instead solves Henderson's
    mixed model equations (Montesinos Lopez et al., equation 2.2)

    .. math::
       \begin{pmatrix}
         X^{T}R^{-1}X & X^{T}R^{-1}Z \\
         Z^{T}R^{-1}X & Z^{T}R^{-1}Z + \Sigma^{-1}
       \end{pmatrix}
       \begin{pmatrix}\hat\beta \\ \hat u\end{pmatrix}
       =
       \begin{pmatrix}X^{T}R^{-1}y \\ Z^{T}R^{-1}y\end{pmatrix},

    whose :math:`\hat\beta` is the BLUE and whose :math:`\hat u` is the
    BLUP. The two routes agree exactly, and ``mme_matches_gls`` reports
    the discrepancy rather than asserting it -- that identity is the
    cheapest available check that the variance components were assembled
    correctly.

    Henderson's form is not merely an alternative derivation. It
    inverts a :math:`(p+q) \times (p+q)` matrix instead of the
    :math:`n \times n` matrix :math:`V`, which is what makes the model
    tractable when :math:`n` is large -- the usual case in genomic
    prediction, where :math:`V` may be tens of thousands square.

    ESTIMABILITY is checked rather than assumed. When :math:`X` is rank
    deficient, :math:`\beta` is not identified and only certain linear
    combinations are. :math:`K^{T}\beta` is estimable if and only if
    :math:`K^{T}(X^{T}X)^{-}(X^{T}X) = K^{T}`, and passing ``K`` runs
    that test. Without it, a rank-deficient design still returns a
    :math:`\hat\beta` -- one of infinitely many -- and ``rank_deficient``
    is the warning that its individual entries mean nothing.

    Parameters
    ----------
    y : array-like, shape (n,)
    X : array-like, shape (n, p)
        Fixed-effects design.
    V : array-like, shape (n, n), optional
        Marginal variance. Identity by default (ordinary least squares).
        Ignored when ``Z``, ``Sigma`` and ``R`` are given.
    Z : array-like, shape (n, q), optional
        Random-effects design.
    Sigma : array-like, shape (q, q), optional
        Variance of the random effects.
    R : array-like, shape (n, n), optional
        Residual variance. Identity by default when ``Z`` is given.
    K : array-like, shape (p, m), optional
        Columns are the linear combinations to test for estimability.

    Returns
    -------
    RichResult
        ``estimate`` / ``beta``, ``se``, ``cov_beta``, ``blup`` and
        ``blup_shrinkage`` when random effects were supplied,
        ``mme_matches_gls``, ``rank_deficient``, ``estimable``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022), *Multivariate
    Statistical Machine Learning Methods for Genomic Prediction*,
    Springer, section 2.2, equations (2.1)-(2.2), pp. 35-37.
    Aitken (1935), *Proceedings of the Royal Society of Edinburgh*
    55:42-48. Henderson (1975), *Biometrics* 31:423-447.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.column_stack([np.ones(4), [0.0, 1.0, 2.0, 3.0]])
    >>> y = np.array([1.0, 3.0, 5.0, 7.0])
    >>> [round(float(b), 6) for b in blue_gls(y, X)["beta"]]
    [1.0, 2.0]
    """
    yv = np.asarray(y, dtype=float).ravel()
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    n = yv.size
    if Xa.shape[0] != n:
        Xa = Xa.T
    if Xa.shape[0] != n:
        raise ValueError(
            "X has %d rows for %d observations." % (Xa.shape[0], n)
        )
    p = Xa.shape[1]

    rank = int(np.linalg.matrix_rank(Xa))
    deficient = rank < p

    use_mme = Z is not None and Sigma is not None
    blup = None
    shrink = None
    mme_gap = None

    if use_mme:
        Za = np.atleast_2d(np.asarray(Z, dtype=float))
        if Za.shape[0] != n:
            Za = Za.T
        q = Za.shape[1]
        Sg = np.atleast_2d(np.asarray(Sigma, dtype=float))
        if Sg.shape != (q, q):
            raise ValueError(
                "Sigma must be %d by %d, got %s." % (q, q, Sg.shape)
            )
        Rm = np.eye(n) if R is None else np.atleast_2d(
            np.asarray(R, dtype=float)
        )
        if Rm.shape != (n, n):
            raise ValueError(
                "R must be %d by %d, got %s." % (n, n, Rm.shape)
            )
        Ri = np.linalg.pinv(Rm)
        Si = np.linalg.pinv(Sg)
        top = np.hstack([Xa.T @ Ri @ Xa, Xa.T @ Ri @ Za])
        bot = np.hstack([Za.T @ Ri @ Xa, Za.T @ Ri @ Za + Si])
        C = np.vstack([top, bot])
        rhs = np.concatenate([Xa.T @ Ri @ yv, Za.T @ Ri @ yv])
        sol = np.linalg.pinv(C) @ rhs
        beta, blup = sol[:p], sol[p:]
        Vm = Za @ Sg @ Za.T + Rm
        # the identity: Henderson's beta-hat IS the GLS estimator
        Vi = np.linalg.pinv(Vm)
        beta_gls = np.linalg.pinv(Xa.T @ Vi @ Xa) @ (Xa.T @ Vi @ yv)
        mme_gap = float(np.max(np.abs(beta - beta_gls)))
        cov = np.linalg.pinv(Xa.T @ Vi @ Xa)
        # shrinkage: BLUP pulls toward zero relative to treating u as fixed
        denom = float(np.linalg.norm(Za.T @ Ri @ (yv - Xa @ beta)))
        shrink = (float(np.linalg.norm(blup)) / denom) if denom > 0 else np.nan
    else:
        Vm = np.eye(n) if V is None else np.atleast_2d(
            np.asarray(V, dtype=float)
        )
        if Vm.shape != (n, n):
            raise ValueError(
                "V must be %d by %d, got %s." % (n, n, Vm.shape)
            )
        Vi = np.linalg.pinv(Vm)
        XtVi = Xa.T @ Vi
        cov = np.linalg.pinv(XtVi @ Xa)
        beta = cov @ (XtVi @ yv)

    estimable = None
    if K is not None:
        Ka = np.atleast_2d(np.asarray(K, dtype=float))
        if Ka.shape[0] != p:
            Ka = Ka.T
        if Ka.shape[0] != p:
            raise ValueError(
                "K must have %d rows, one per fixed effect." % p
            )
        XtX = Xa.T @ Xa
        H = np.linalg.pinv(XtX) @ XtX
        estimable = np.array([
            bool(np.allclose(Ka[:, j] @ H, Ka[:, j], atol=1e-8))
            for j in range(Ka.shape[1])
        ])

    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return RichResult(
        payload={
            "estimate": beta,
            "beta": beta,
            "se": se,
            "cov_beta": cov,
            "blup": blup,
            "blup_shrinkage": shrink,
            "blup_note": (
                None if blup is None else
                "the BLUP shrinks toward zero; that shrinkage is the point "
                "of treating the effect as random rather than fixed"
            ),
            "mme_matches_gls": mme_gap,
            "mme_note": (
                None if mme_gap is None else
                "largest absolute difference between Henderson's beta-hat "
                "and the GLS estimator; they are the same quantity, so a "
                "non-trivial value means the variance components were "
                "assembled wrongly"
            ),
            "rank": rank,
            "rank_deficient": bool(deficient),
            "rank_note": (
                None if not deficient else
                "X is rank deficient, so beta is not identified and the "
                "individual entries returned are one solution of infinitely "
                "many; only estimable functions K'beta are meaningful"
            ),
            "estimable": estimable,
            "n": int(n),
            "p": int(p),
            "method": ("BLUE and BLUP from Henderson's mixed model equations"
                       if use_mme else
                       "Generalised least squares (BLUE)"),
        }
    )


def cheatsheet():
    return (
        "bluerg: GLS/BLUE, or Henderson's MME for BLUE and BLUP together, "
        "with the MME-equals-GLS identity and an estimability test"
    )
