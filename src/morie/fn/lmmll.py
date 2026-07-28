# morie.fn -- function file (rootcoder007/morie)
"""Linear mixed model log-likelihood."""

import numpy as np

from ._richresult import RichResult

__all__ = ["lmm_loglik"]


def lmm_loglik(y, X, Z=None, D=None, R=None, V=None, reml=False):
    r"""Gaussian log-likelihood of a linear mixed model.

    Montesinos Lopez et al. equation (5.2):

    .. math::
       L(\beta, D, R; y) = \frac{|V|^{-1/2}}{(2\pi)^{n/2}}
         \exp\left[-\tfrac12 (y - X\beta)' V^{-1}(y - X\beta)\right],
       \qquad V = ZDZ' + R.

    The book's text writes :math:`V \equiv Z'DZ + R` one line below
    stating the marginal as :math:`N(X\beta, ZDZ' + R)`. Only the
    latter is dimensionally possible -- with :math:`Z` of order
    :math:`n\times q`, :math:`Z'DZ` is :math:`q\times q` and cannot be
    the variance of an :math:`n`-vector -- so :math:`ZDZ'` is what is
    implemented.

    ``reml=True`` adds :math:`\ln|X'V^{-1}X|` and drops :math:`k` from
    the constant. That term is the entire difference between the two
    objectives and it is what removes the downward bias of the ML
    variance estimate. The consequence is that REML values are NOT
    comparable across different mean structures -- they are likelihoods
    of different data, the error contrasts rather than :math:`y` --
    and ``comparable_across_mean_models`` records which regime applies.

    Parameters
    ----------
    y : array-like, shape (n,)
    X : array-like, shape (n, p)
    Z : array-like, shape (n, q), optional
    D : array-like, shape (q, q), optional
    R : array-like, shape (n, n), optional
        Residual variance; :math:`\sigma^2 I` is not assumed.
    V : array-like, shape (n, n), optional
        Supply the marginal variance directly instead of Z, D, R.
    reml : bool

    Returns
    -------
    RichResult
        ``loglik``, ``neg2loglik``, ``beta``, ``aic``, ``bic``,
        ``logdet_V``, ``quadratic_form``, ``comparable_across_mean_models``.

    References
    ----------
    Montesinos Lopez, Montesinos Lopez and Crossa (2022),
    *Multivariate Statistical Machine Learning Methods for Genomic
    Prediction*, Springer, section 5.2.1, equation (5.2), pp. 141-143.
    Patterson and Thompson (1971) for REML.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.ones((5, 1))
    >>> out = lmm_loglik(np.arange(5.0), X, V=np.eye(5))
    >>> bool(out["loglik"] < 0)
    True
    """
    yv = np.asarray(y, dtype=float).ravel()
    Xa = np.atleast_2d(np.asarray(X, dtype=float))
    n = yv.size
    if Xa.shape[0] != n:
        Xa = Xa.T
    if Xa.shape[0] != n:
        raise ValueError("X has %d rows for %d observations."
                         % (Xa.shape[0], n))
    p = Xa.shape[1]

    if V is None:
        if Z is None or D is None:
            raise ValueError("supply V, or both Z and D.")
        Za = np.atleast_2d(np.asarray(Z, dtype=float))
        if Za.shape[0] != n:
            Za = Za.T
        q = Za.shape[1]
        Dm = np.atleast_2d(np.asarray(D, dtype=float))
        if Dm.shape != (q, q):
            raise ValueError("D must be %d by %d, got %s." % (q, q, Dm.shape))
        Rm = np.eye(n) if R is None else np.atleast_2d(
            np.asarray(R, dtype=float)
        )
        if Rm.shape != (n, n):
            raise ValueError("R must be %d by %d, got %s." % (n, n, Rm.shape))
        Vm = Za @ Dm @ Za.T + Rm          # ZDZ', not Z'DZ
    else:
        Vm = np.atleast_2d(np.asarray(V, dtype=float))
        if Vm.shape != (n, n):
            raise ValueError("V must be %d by %d, got %s." % (n, n, Vm.shape))

    Vm = 0.5 * (Vm + Vm.T)
    sign, logdet = np.linalg.slogdet(Vm)
    if sign <= 0:
        raise ValueError(
            "V is not positive definite; the variance components are "
            "inadmissible."
        )
    Vi = np.linalg.inv(Vm)
    ViX = Vi @ Xa
    XtViX = Xa.T @ ViX
    beta = np.linalg.solve(XtViX, ViX.T @ yv)
    r = yv - Xa @ beta
    quad = float(r @ Vi @ r)

    if reml:
        s2, ld2 = np.linalg.slogdet(XtViX)
        if s2 <= 0:
            raise ValueError("X'V^{-1}X is singular; X is rank deficient.")
        ll = -0.5 * (logdet + ld2 + quad + (n - p) * np.log(2 * np.pi))
        k = p
    else:
        ll = -0.5 * (logdet + quad + n * np.log(2 * np.pi))
        ld2 = None
        k = 0
    npar = p + 1
    return RichResult(
        payload={
            "estimate": float(ll),
            "loglik": float(ll),
            "neg2loglik": float(-2 * ll),
            "beta": beta,
            "residuals": r,
            "logdet_V": float(logdet),
            "logdet_XtViX": None if ld2 is None else float(ld2),
            "quadratic_form": quad,
            "reml": bool(reml),
            "reml_note": (
                "the ln|X'V^-1 X| term is the whole difference; it accounts "
                "for the degrees of freedom spent on the mean and removes "
                "the downward bias of the ML variance estimate"
            ),
            "comparable_across_mean_models": not reml,
            "comparison_note": (
                "REML values are likelihoods of the error contrasts, not of "
                "y, so they may not be compared across different mean "
                "structures; ML values may"
            ),
            "aic": float(-2 * ll + 2 * npar),
            "bic": float(-2 * ll + npar * np.log(n)),
            "variance_note": (
                "V = ZDZ' + R. The book's text writes Z'DZ one line below "
                "giving the marginal as N(Xb, ZDZ' + R); only ZDZ' has the "
                "right dimension for the variance of an n-vector"
            ),
            "n": int(n),
            "p": int(p),
            "method": "%s log-likelihood of a linear mixed model"
                      % ("REML" if reml else "ML"),
        }
    )


def cheatsheet():
    return (
        "lmmll: LMM ML/REML log-likelihood with the ZDZ' correction and the "
        "likelihood-comparison rule"
    )
