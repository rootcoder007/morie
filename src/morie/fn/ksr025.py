# morie.fn -- function file (rootcoder007/morie)
"""Penalised log-likelihood."""

import numpy as np

from ._richresult import RichResult

__all__ = ["kosorok_ch1_penalized_loglikelihood"]


def kosorok_ch1_penalized_loglikelihood(loglik_terms, J_eta, lambda_n, n=None,
                                        beta=None, eta=None, X=None):
    r"""Penalised log-likelihood criterion (Kosorok Ch. 1):

    .. math:: \tilde L_n(\beta, \eta) = n^{-1}\sum_i
              \log p_{\beta,\eta}(X_i) - \lambda_n^2 J^2(\eta).

    Note the penalty enters as :math:`\lambda_n^2 J^2` -- BOTH
    squared. The smoothing parameter must shrink at a rate tied to n
    for the estimator to be consistent yet root-n efficient in beta;
    a fixed lambda leaves an asymptotic bias, which is why lambda
    carries the subscript n.

    Parameters
    ----------
    loglik_terms : array-like, shape (n,)
        Per-observation log-likelihood contributions.
    J_eta : float
        The roughness functional J(eta).
    lambda_n : float
        Smoothing parameter, non-negative.
    n : int, optional
        Sample size; taken from loglik_terms.
    beta, eta, X : ignored
        Interface compatibility.

    Returns
    -------
    RichResult
        keys: ``criterion``, ``mean_loglik``, ``penalty``,
        ``lambda_n``, ``J_eta``, ``n``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 1 (penalised likelihood estimation).
    """
    ll = np.asarray(loglik_terms, dtype=float).ravel()
    if ll.size < 1:
        raise ValueError("loglik_terms must be non-empty.")
    if n is not None and int(n) != ll.size:
        raise ValueError(f"n = {n} does not match len(loglik_terms) = {ll.size}.")
    lam = float(lambda_n)
    if lam < 0:
        raise ValueError(f"lambda_n must be non-negative, got {lam}.")
    J = float(J_eta)
    if J < 0:
        raise ValueError(f"J_eta must be non-negative, got {J}.")
    mean_ll = float(np.mean(ll))
    pen = lam**2 * J**2
    return RichResult(
        payload={"criterion": mean_ll - pen, "mean_loglik": mean_ll,
                 "penalty": pen, "lambda_n": lam, "J_eta": J, "n": int(ll.size),
                 "method": "n^-1 sum log p - lambda_n^2 J^2(eta) (both squared)"}
    )


def cheatsheet():
    return "ksr025: penalty is lambda^2 J^2, BOTH squared; lambda must shrink with n"
