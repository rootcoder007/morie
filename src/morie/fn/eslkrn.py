# morie.fn -- function file (rootcoder007/morie)
"""Parzen-Rosenblatt kernel density estimate, ESL Sec. 6.6."""

import numpy as np

from ._richresult import RichResult

__all__ = ["esl_kernel_density"]


def esl_kernel_density(x, data, lambda_=None):
    r"""Gaussian kernel density estimate, ESL Eqs. (6.23)-(6.24):

    .. math:: \hat f_X(x) = \frac1N\sum_{i=1}^N \phi_\lambda(x - x_i)
              = (\hat F \star \phi_\lambda)(x),

    and in :math:`\mathbb R^p`, with the Gaussian product kernel,

    .. math:: \hat f_X(x_0) = \frac1{N(2\lambda^2\pi)^{p/2}}
              \sum_{i=1}^N e^{-\frac12(\|x_i - x_0\|/\lambda)^2}.

    The book's framing is worth keeping: (6.23) is the CONVOLUTION of
    the empirical distribution with a Gaussian of standard deviation
    :math:`\lambda`. :math:`\hat F` puts mass ``1/N`` at each
    observation and is jumpy; the estimate smooths it by adding
    independent Gaussian noise to each :math:`x_i`. That is also why
    :math:`\lambda` is a standard deviation here rather than a
    generic "bandwidth" -- it is the noise being added.

    Parameters
    ----------
    x : array-like
        Evaluation points; shape (m,) in one dimension or (m, p).
    data : array-like
        Sample, shape (N,) or (N, p).
    lambda_ : float, optional
        Kernel standard deviation. Silverman's normal reference
        ``1.06 sigma N^{-1/5}`` when omitted, which is a rule for
        DENSITY estimation and correct here.

    Returns
    -------
    RichResult
        keys: ``density``, ``x``, ``lambda``, ``n``, ``p``,
        ``is_convolution``, ``normaliser``, ``mass`` (1-D only),
        ``method``.

    References
    ----------
    Hastie, Tibshirani and Friedman, *The Elements of Statistical
    Learning*, 2nd ed., Sec. 6.6, Eqs. (6.22)-(6.24). Read from the
    PDF. Parzen (1962); Loader (1999) for the log-scale refinements
    the book mentions but does not pursue.
    """
    from ._esl import gaussian_product_kernel_density

    D = np.atleast_2d(np.asarray(data, dtype=float))
    if D.shape[0] == 1 and D.shape[1] > 1:
        D = D.T
    N, p = D.shape
    if N < 2:
        raise ValueError(f"need at least 2 observations, got {N}.")
    if lambda_ is None:
        s = float(np.std(D[:, 0], ddof=1))
        lam = 1.06 * (s if s > 0 else 1.0) * N ** (-0.2)
    else:
        lam = float(lambda_)
    if lam <= 0:
        raise ValueError(f"lambda must be positive, got {lam}.")
    Q = np.atleast_2d(np.asarray(x, dtype=float))
    if Q.shape[0] == 1 and Q.shape[1] != p:
        Q = Q.reshape(-1, p) if Q.size % p == 0 else Q.T
    dens = gaussian_product_kernel_density(Q, D, lam)
    mass = None
    if p == 1 and Q.shape[0] > 2:
        g = Q.ravel()
        if np.all(np.diff(g) > 0):
            mass = float(np.trapezoid(dens, g))
    return RichResult(payload={
        "x": Q if p > 1 else Q.ravel(), "density": dens, "lambda": lam,
        "n": int(N), "p": int(p), "mass": mass,
        "normaliser": float(N * (2.0 * lam ** 2 * np.pi) ** (p / 2.0)),
        "is_convolution": True,
        "convolution_note": "(6.23): the empirical df convolved with a "
                            "Gaussian of standard deviation lambda",
        "method": "ESL (6.23)/(6.24) Parzen density with a Gaussian product kernel"})


def cheatsheet():
    return "eslkrn: (6.23) is F_hat convolved with a Gaussian -- lambda IS the added noise's sd"
