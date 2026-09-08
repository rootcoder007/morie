# morie.fn -- internal helpers (rootcoder007/morie)
"""Shared machinery for the instrumental-variables and
doubly-robust modules.

The estimators here are all identification results first and
formulas second, and the formulas are short enough that the
temptation is to write them down and move on. The conditions are
what the tests check: a Wald ratio with a weak first stage is a
number divided by noise, a k-class estimator with the wrong k is a
different estimator entirely, and a cross-fitted score that was not
actually cross-fitted carries the regularisation bias it exists to
remove.
"""

from . import _array_core as np

__all__ = ["add_intercept", "projection", "annihilator", "k_class",
           "first_stage_f", "folds"]


def add_intercept(X):
    """Prepend a column of ones unless one is already there."""
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.size and np.any(np.all(np.isclose(A, 1.0), axis=0)):
        return A
    return np.column_stack([np.ones(A.shape[0]), A])


def projection(Z, M):
    r""":math:`P_Z M = Z(Z'Z)^{-}Z'M`, by least squares rather than a
    formed inverse -- ``Z'Z`` is routinely near-singular when
    instruments are correlated, and inverting it explicitly turns a
    warning into a wrong answer."""
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    M = np.asarray(M, dtype=float)
    coef = np.linalg.lstsq(Z, M, rcond=None)[0]
    return Z @ coef


def annihilator(Z, M):
    r""":math:`M_Z M = (I - P_Z)M`, the residual maker."""
    return np.asarray(M, dtype=float) - projection(Z, M)


def k_class(y, X, Z, k):
    r"""The k-class estimator

    .. math:: \hat\beta_k = \left(X'(I - k M_Z)X\right)^{-1}
              X'(I - k M_Z)y ,

    with :math:`M_Z = I - Z(Z'Z)^{-1}Z'`.

    The family is worth keeping in one place because the special
    cases are exactly the estimators in this shelf: ``k = 0`` is
    ordinary least squares, ``k = 1`` is two-stage least squares, and
    ``k`` equal to the smallest eigenvalue of the Anderson-Rubin
    variance ratio is limited-information maximum likelihood. Writing
    each separately invites them to drift apart.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    MzX = annihilator(Z, X)
    Mzy = annihilator(Z, y)
    A = X.T @ X - k * (X.T @ MzX)
    b = X.T @ y - k * (X.T @ Mzy)
    return np.linalg.lstsq(A, b, rcond=None)[0]


def first_stage_f(D, Z, exog=None):
    r"""The first-stage F statistic for the excluded instruments.

    The single most useful diagnostic in this shelf. Every estimator
    here divides by a first-stage association, so when that
    association is weak the estimator is a ratio of two noisy
    quantities and its sampling distribution is not the normal one
    the standard error describes. The conventional rule of thumb is
    that F below about 10 is a warning; it is a rule of thumb and not
    a threshold.
    """
    D = np.asarray(D, dtype=float).ravel()
    Zf = add_intercept(Z) if exog is None else np.column_stack(
        [add_intercept(exog), np.atleast_2d(np.asarray(Z, dtype=float))])
    W = add_intercept(exog) if exog is not None else np.ones((D.size, 1))
    rss_r = float(np.sum(annihilator(W, D) ** 2))
    rss_u = float(np.sum(annihilator(Zf, D) ** 2))
    q = Zf.shape[1] - W.shape[1]
    dfr = D.size - Zf.shape[1]
    if q < 1 or dfr < 1 or rss_u <= 0:
        return float("nan")
    return float(((rss_r - rss_u) / q) / (rss_u / dfr))


def folds(n, n_folds, seed=0):
    """Index sets for K-fold cross-fitting, shuffled."""
    n_folds = int(n_folds)
    if not 2 <= n_folds <= n:
        raise ValueError(
            f"n_folds must lie in 2..{n}, got {n_folds}; cross-fitting with "
            "one fold is not cross-fitting.")
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::n_folds] for i in range(n_folds)]


def cheatsheet():
    return ("_caus_iv: every estimator here divides by a first-stage "
            "association -- check F before believing the standard error")
