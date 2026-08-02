# morie.fn -- function file (rootcoder007/morie)
"""Sample standard deviation."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_sample_std"]


def rangayyan_ch3_sample_std(eta, mu_eta=None, N=None):
    r"""Sample standard deviation (Rangayyan Ch. 3):

    .. math:: \sigma_\eta = \sqrt{\frac1N \sum_{n=0}^{N-1}
              (\eta(n) - \mu_\eta)^2}.

    The book's divisor is N, not N - 1: this is the second central
    moment of the observed record, not an unbiased estimator of a
    population variance. Both are returned, since substituting one
    for the other is a common slip.

    Parameters
    ----------
    eta : array-like
        Samples.
    mu_eta : float, optional
        Mean to centre on; the sample mean if omitted.
    N : int, optional
        Length.

    Returns
    -------
    RichResult
        keys: ``std`` (divisor N), ``std_unbiased`` (divisor N - 1),
        ``mean_used``, ``N``, ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3.
    """
    eta = np.asarray(eta, dtype=float).ravel()
    if eta.size < 1:
        raise ValueError("eta must be non-empty.")
    if N is not None and int(N) != eta.size:
        raise ValueError(f"N = {N} does not match len(eta) = {eta.size}.")
    mu = float(np.mean(eta)) if mu_eta is None else float(mu_eta)
    dev2 = (eta - mu) ** 2
    n = eta.size
    unb = float(np.sqrt(dev2.sum() / (n - 1))) if n > 1 else float("nan")
    return RichResult(payload={"std": float(np.sqrt(dev2.mean())),
                               "std_unbiased": unb, "mean_used": mu, "N": int(n),
                               "method": "sigma with divisor N (the book's convention)"})


def cheatsheet():
    return "rng010: divisor N, not N-1; both returned"
