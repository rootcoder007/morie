# morie.fn -- function file (rootcoder007/morie)
"""Newey-West HAC covariance estimator."""

import numpy as np

from ._richresult import RichResult

__all__ = ["newey_west_hac"]


def newey_west_hac(scores, lags=None, X=None, prewhiten=False):
    r"""Heteroskedasticity- and autocorrelation-consistent covariance.

    .. math::
       \hat\Omega = \hat\Gamma_0
         + \sum_{j=1}^{L} w_j\big(\hat\Gamma_j + \hat\Gamma_j'\big),
       \qquad w_j = 1 - \frac{j}{L+1}

    The Bartlett weights are not a smoothing convenience. Truncating
    the sum at lag :math:`L` with EQUAL weights can produce a matrix
    that is not positive semi-definite -- a "variance" estimate with a
    negative eigenvalue, which yields imaginary standard errors. The
    linearly declining weights are what guarantee positive
    semi-definiteness, and that is the whole reason for the kernel.

    Choosing :math:`L` trades bias against variance and there is no
    free answer. Too small and residual autocorrelation is left in, so
    the standard errors are too small; too large and the estimate is
    itself noisy. The default follows Newey and West's automatic rule
    :math:`L = \lfloor 4(T/100)^{2/9}\rfloor`, which is reported rather
    than hidden.

    ``inflation`` gives the ratio of the HAC standard errors to the
    naive ones. Under positive autocorrelation, which is the usual
    case in time series, it exceeds 1 -- the amount by which ordinary
    standard errors were overstating precision.

    Parameters
    ----------
    scores : array-like, shape (T,) or (T, k)
        Moment contributions, typically :math:`x_t \hat u_t`.
    lags : int, optional
    X : array-like, optional
        Regressor matrix; supplying it returns the sandwich covariance
        of the coefficients rather than of the moments.
    prewhiten : bool
        Fit a VAR(1) first and recolour afterwards.

    Returns
    -------
    RichResult
        ``covariance``, ``se``, ``lags``, ``inflation``,
        ``positive_definite``, ``autocorrelation``.

    References
    ----------
    Newey and West (1987), *Econometrica* 55:703-708.
    Newey and West (1994), *Review of Economic Studies* 61:631-653,
    for the automatic lag rule.
    Andrews and Monahan (1992) for prewhitening.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> out = newey_west_hac(rng.normal(size=(200, 1)), lags=4)
    >>> bool(out["positive_definite"])
    True
    """
    S = np.atleast_2d(np.asarray(scores, dtype=float))
    if S.shape[0] == 1 and S.shape[1] > 1:
        S = S.T
    T, k = S.shape
    if T < 3:
        raise ValueError("need at least 3 observations, got %d." % T)
    L = (int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
         if lags is None else int(lags))
    L = max(min(L, T - 1), 0)

    A = None
    W = S - S.mean(axis=0, keepdims=True)
    if prewhiten and T > k + 2:
        Y, Xp = W[1:], W[:-1]
        A = np.linalg.lstsq(Xp, Y, rcond=None)[0]
        ev = np.max(np.abs(np.linalg.eigvals(A))) if k > 1 else abs(float(A))
        if ev < 0.97:
            W = Y - Xp @ A
            T = W.shape[0]
        else:
            A = None                      # too close to a unit root to whiten

    G0 = W.T @ W / T
    Om = G0.copy()
    acf = []
    for j in range(1, L + 1):
        Gj = W[j:].T @ W[:-j] / T
        w = 1.0 - j / (L + 1.0)
        Om = Om + w * (Gj + Gj.T)
        d = np.sqrt(np.maximum(np.diag(G0), 1e-300))
        acf.append(float(np.mean(np.diag(Gj) / (d * d))))
    if A is not None:
        M = np.linalg.pinv(np.eye(k) - A)
        Om = M @ Om @ M.T

    naive = np.sqrt(np.maximum(np.diag(G0) / T, 0.0))
    if X is not None:
        Xa = np.atleast_2d(np.asarray(X, dtype=float))
        if Xa.shape[0] != S.shape[0]:
            Xa = Xa.T
        bread = np.linalg.pinv(Xa.T @ Xa / Xa.shape[0])
        cov = bread @ Om @ bread / Xa.shape[0]
    else:
        cov = Om / T
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    eig = np.linalg.eigvalsh((Om + Om.T) / 2.0)
    return RichResult(
        payload={
            "estimate": cov,
            "covariance": cov,
            "se": se,
            "omega": Om,
            "lags": int(L),
            "lag_rule": ("supplied" if lags is not None
                         else "Newey-West automatic, floor(4 (T/100)^(2/9))"),
            "lag_note": (
                "too few lags leaves autocorrelation in and understates the "
                "standard errors; too many makes the estimate itself noisy"
            ),
            "bartlett_note": (
                "the linearly declining weights are what guarantee positive "
                "semi-definiteness; a flat truncation can return a matrix "
                "with a negative eigenvalue and hence imaginary standard "
                "errors"
            ),
            "inflation": (se / np.maximum(naive, 1e-300)
                          if X is None else None),
            "inflation_note": (
                "HAC standard errors divided by the naive ones; above 1 "
                "under positive autocorrelation, which is how much ordinary "
                "errors were overstating precision"
            ),
            "autocorrelation": np.asarray(acf),
            "min_eigenvalue": float(eig.min()),
            "positive_definite": bool(eig.min() > -1e-10),
            "prewhitened": bool(A is not None),
            "T": int(S.shape[0]),
            "k": int(k),
            "method": "Newey-West HAC covariance",
        }
    )


def cheatsheet():
    return (
        "nwest: Bartlett-kernel HAC covariance with the automatic lag rule "
        "and the PSD guarantee the weights exist for"
    )
