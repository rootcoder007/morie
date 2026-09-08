# morie.fn -- function file (rootcoder007/morie)
"""Kozachenko-Leonenko k-NN entropy estimator."""

import math

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["knn_entropy", "entropy_knn"]


def _digamma(x):
    """Digamma by recurrence plus asymptotic series."""
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return (r + math.log(x) - 0.5 / x
            + f * (-1 / 12.0 + f * (1 / 120.0 + f * (-1 / 252.0
                   + f * (1 / 240.0 + f * (-1 / 132.0))))))


def knn_entropy(x, k=3, base="nats"):
    r"""Differential entropy from nearest-neighbour distances.

    .. math::
       \hat H = -\psi(k) + \psi(n) + \log c_d
                + \frac{d}{n}\sum_{i=1}^{n}\log \varepsilon_i

    with :math:`\varepsilon_i` the distance to the :math:`k`-th nearest
    neighbour and :math:`c_d` the volume of the unit :math:`d`-ball.

    The :math:`-\psi(k)` term is what makes this an entropy estimator
    rather than a plug-in. Naively one would use :math:`\log k` --
    treating the :math:`k`-ball as containing a fraction :math:`k/n` of
    the mass -- and that is biased, because the number of points in a
    fixed-radius ball is Poisson and :math:`E[\log N] \ne \log E[N]`.
    The digamma is the exact correction, and it is why the estimator is
    asymptotically unbiased for any continuous density.

    Two limits are worth stating. Differential entropy is NOT
    invariant to rescaling -- multiplying :math:`x` by :math:`a` adds
    :math:`d\log a` -- so it is not comparable across units and can be
    negative. And the estimator degrades badly in high dimension,
    where nearest-neighbour distances concentrate and all points look
    equidistant; ``distance_concentration`` reports the ratio of the
    spread of :math:`\varepsilon` to its mean, which collapses toward
    zero exactly when the estimate stops meaning anything.

    Parameters
    ----------
    x : array-like, shape (n,) or (n, d)
    k : int
        Neighbour rank.
    base : {'nats', 'bits'}

    Returns
    -------
    RichResult
        ``entropy``, ``k``, ``dimension``, ``distance_concentration``,
        ``gaussian_reference``, ``negentropy``.

    References
    ----------
    Kozachenko and Leonenko (1987), *Problems of Information
    Transmission* 23:95-101.
    Kraskov, Stogbauer and Grassberger (2004), *Physical Review E*
    69:066138.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> h = knn_entropy(rng.normal(size=2000))["entropy"]
    >>> bool(abs(h - 0.5 * np.log(2 * np.pi * np.e)) < 0.1)
    True
    """
    X = np.atleast_2d(np.asarray(x, dtype=float))
    if X.shape[0] == 1 and X.shape[1] > 1:
        X = X.T
    n, d = X.shape
    k = int(k)
    if k < 1:
        raise ValueError("k must be at least 1, got %d." % k)
    if n <= k:
        raise ValueError(
            "need more than k = %d observations, got %d." % (k, n)
        )
    if base not in ("nats", "bits"):
        raise ValueError("base must be 'nats' or 'bits', got %r." % base)

    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    eps = np.sort(D, axis=1)[:, k - 1]
    if np.any(eps <= 0):
        raise ValueError(
            "duplicate points give a zero neighbour distance; jitter the "
            "data or lower k."
        )
    log_cd = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0)
    H = (-_digamma(k) + _digamma(n) + log_cd
         + (d / n) * float(np.sum(np.log(eps))))
    if base == "bits":
        H = H / math.log(2.0)

    cov = np.cov(X, rowvar=False)
    cov = np.atleast_2d(cov)
    sign, ld = np.linalg.slogdet(cov + 1e-12 * np.eye(d))
    gauss = 0.5 * (d * math.log(2 * math.pi * math.e) + ld) if sign > 0 \
        else np.nan
    if base == "bits" and gauss == gauss:
        gauss = gauss / math.log(2.0)
    return RichResult(
        payload={
            "estimate": float(H),
            "entropy": float(H),
            "base": base,
            "k": k,
            "dimension": int(d),
            "neighbour_distances": eps,
            # ddof=1 to match R's stats::sd; numpy defaults to the
            # population divisor and the two differ by sqrt(n/(n-1)),
            # which is enough to break a ten-digit parity anchor
            "distance_concentration": float(
                np.std(eps, ddof=1) / np.mean(eps)
            ),
            "concentration_note": (
                "spread of the k-th neighbour distance over its mean; as "
                "this collapses toward zero the points are all equidistant "
                "and the estimate stops carrying information"
            ),
            "gaussian_reference": float(gauss),
            "negentropy": float(gauss - H) if gauss == gauss else np.nan,
            "negentropy_note": (
                "entropy of a Gaussian with the same covariance minus the "
                "estimate; non-negative in population, and a measure of how "
                "far from Gaussian the data are"
            ),
            "digamma_note": (
                "the -psi(k) term corrects for E[log N] != log E[N] when the "
                "count in a ball is Poisson; using log k instead leaves a "
                "bias that does not vanish"
            ),
            "scale_note": (
                "differential entropy is not scale invariant -- rescaling x "
                "by a adds d log a -- so it is not comparable across units "
                "and may be negative"
            ),
            "n": int(n),
            "method": "Kozachenko-Leonenko k-NN differential entropy",
        }
    )


def cheatsheet():
    return (
        "entest: k-NN differential entropy with the digamma correction and "
        "a high-dimension concentration warning"
    )


#: Catalogue alias for :func:`knn_entropy`.
entropy_knn = knn_entropy


# compact alias per ledger/NAMING.md
entropyknn = entropy_knn


# compact alias per ledger/NAMING.md
knnentropy = knn_entropy
