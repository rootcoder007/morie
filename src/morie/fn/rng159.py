# morie.fn -- function file (rootcoder007/morie)
"""LMS gradient estimate."""

import numpy as np

from ._richresult import RichResult

__all__ = ["rangayyan_ch3_lms_gradient_estimate"]


def rangayyan_ch3_lms_gradient_estimate(r, e, x=None, w=None, n=None):
    r"""LMS instantaneous gradient estimate (Rangayyan Ch. 3):

    .. math:: \widehat{\nabla}\,e^2(n) = -2 e(n)\, \mathbf{r}(n).

    Widrow-Hoff's key simplification: the true gradient of the MEAN
    squared error needs an expectation, but the gradient of the
    INSTANTANEOUS squared error needs only the current sample. The
    algorithm is a stochastic gradient descent whose noisy steps
    average to the right direction, which is why LMS converges in the
    mean rather than monotonically.

    Parameters
    ----------
    r : array-like, shape (N, p)
        Reference vectors.
    e : array-like, shape (N,)
        Instantaneous errors.
    x, w : ignored
        Interface compatibility -- the identity
        -2xr + 2(w'r)r = -2er already folds them in.
    n : int, optional
        Index to report.

    Returns
    -------
    RichResult
        keys: ``gradient`` (N, p), ``gradient_at_n``, ``N``, ``p``,
        ``method``.
    References
    ----------
    Rangayyan, R. M. (2015). *Biomedical Signal Analysis* (2nd ed.).
    Wiley-IEEE Press. Ch. 3 (the LMS algorithm).
    """
    R = np.atleast_2d(np.asarray(r, dtype=float))
    ev = np.asarray(e, dtype=float).ravel()
    if R.shape[0] != ev.size:
        R = R.T
    if R.shape[0] != ev.size:
        raise ValueError("r must have one row per entry of e.")
    grad = -2.0 * ev[:, None] * R
    at_n = None
    if n is not None:
        idx = int(n)
        if not 0 <= idx < grad.shape[0]:
            raise ValueError(f"n must lie in 0..{grad.shape[0] - 1}, got {idx}.")
        at_n = grad[idx]
    return RichResult(payload={"gradient": grad, "gradient_at_n": at_n,
                               "N": int(R.shape[0]), "p": int(R.shape[1]),
                               "method": "grad e^2(n) = -2 e(n) r(n); stochastic, not exact"})


def cheatsheet():
    return "rng159: instantaneous gradient needs no expectation -- that IS the trick"
