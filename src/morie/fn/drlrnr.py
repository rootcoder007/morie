# morie.fn -- function file (rootcoder007/morie)
"""R-learner for the CATE (Nie and Wager).

DUPLICATE.  The method this module names -- Nie, X. and Wager, S.
(2021), "Quasi-oracle estimation of heterogeneous treatment effects",
*Biometrika* 108(2), 299-319, doi:10.1093/biomet/asaa076 -- is already
implemented in ``morie.fn.rlear`` as ``rlearn``: the Robinson
decomposition eq. (1) and the R-learner objective eq. (4) with the
regularizer set to zero and the cross-fitted nuisance predictions
supplied by the caller.

Note that ``ledger/wave2/DUPMAP.tsv`` pointed this module at ``slearn``.
That is wrong: ``slearn`` is the S-learner of Kuenzel et al., a single
outcome model with the treatment as a feature, which is a different
estimator.  The correct alias target is ``rlear``, and this file records
the correction.

The argument names map straight across: ``ml_outcome`` is the
cross-fitted m_hat(X) and ``ml_propensity`` is the cross-fitted
e_hat(X), which is exactly what ``rlear.rlearn`` expects.
"""

from __future__ import annotations

from .rlear import rlearn as _impl

__all__ = ["r_learner"]


def r_learner(y, D, X, ml_outcome, ml_propensity):
    """R-learner CATE from caller-supplied cross-fitted nuisances.

    Alias of :func:`morie.fn.rlear.rlearn`.

    Parameters
    ----------
    y : array-like
        Outcome.
    D : array-like
        Treatment indicator W.
    X : 2-D array-like or None
        Basis columns for a linear tau(X); ``None`` fits a constant
        treatment effect.
    ml_outcome : array-like
        Cross-fitted m_hat(X_i) = E(Y | X = X_i).
    ml_propensity : array-like
        Cross-fitted e_hat(X_i) = pr(W = 1 | X = X_i).

    Returns
    -------
    result : dict
        Keys: tau, ate, loss, n.

    References
    ----------
    Nie & Wager (2021), Biometrika 108(2):299-319,
    doi:10.1093/biomet/asaa076.
    """
    return _impl(y=y, t=D, m=ml_outcome, e=ml_propensity, x=X)


def cheatsheet():
    return "drlrnr: R-learner for CATE (Nie-Wager) -- alias of rlear"


# compact alias per ledger/NAMING.md
rlearner = r_learner
