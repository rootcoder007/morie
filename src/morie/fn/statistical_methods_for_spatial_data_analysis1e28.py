# morie.fn -- function file (rootcoder007/morie)
"""AR(1) covariance matrix and its closed-form inverse."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = [
    "ar1cov",
    "statistical_methods_for_spatial_data_analysis_chapter_1_equation_28",
]


def ar1cov(n, rho, sigma2=1.0):
    r"""First-order autoregressive covariance and precision matrix.

    Equation (1.28), p. 35, sets

    .. math::

        Y_i = \mu + e_i, \quad \mathrm{E}[e_i] = 0, \quad
        \mathrm{Cov}[Y_i, Y_j] = \sigma^2\rho^{|i-j|},
        \quad i = 1,\dots,n.

    The inverse is a tridiagonal ("type 2 diagonal") matrix given on the
    same page,

    .. math::

        \boldsymbol\Sigma^{-1} = \frac{1}{\sigma^2(1-\rho^2)}
        \begin{bmatrix}
        1 & -\rho & & \\
        -\rho & 1+\rho^2 & -\rho & \\
        & \ddots & \ddots & \ddots \\
        & & -\rho & 1
        \end{bmatrix},

    with :math:`1+\rho^2` on the interior of the diagonal and 1 at both
    ends.  Building it directly avoids an ``O(n^3)`` inversion.

    Parameters
    ----------
    n : int
        Dimension, at least 2.
    rho : float
        Lag-one correlation, strictly inside ``(-1, 1)``.
    sigma2 : float
        Marginal variance, positive.

    Returns
    -------
    RichResult
        ``sigma``, ``precision``, ``logdet``, ``n``, ``rho``, ``sigma2``.

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC, eq. (1.28), p. 35;
    inverse from Graybill (1983, pp. 198-201).
    """
    n = int(n)
    rho = float(rho)
    sigma2 = float(sigma2)
    if n < 2:
        raise ValueError("`n` must be at least 2")
    if sigma2 <= 0:
        raise ValueError("`sigma2` must be positive")
    if rho <= -1.0 or rho >= 1.0:
        raise ValueError("`rho` must lie strictly inside (-1, 1)")

    sig = [[sigma2 * rho ** abs(i - j) for j in range(n)] for i in range(n)]
    c = 1.0 / (sigma2 * (1.0 - rho * rho))
    prec = [[0.0] * n for _ in range(n)]
    for i in range(n):
        prec[i][i] = c * (1.0 if (i == 0 or i == n - 1) else (1.0 + rho * rho))
        if i + 1 < n:
            prec[i][i + 1] = -c * rho
            prec[i + 1][i] = -c * rho
    # |Sigma| = (sigma^2)^n (1 - rho^2)^(n-1) for the AR(1) structure.
    import math

    logdet = n * math.log(sigma2) + (n - 1) * math.log(1.0 - rho * rho)

    return RichResult(
        title="AR(1) covariance (eq. 1.28)",
        summary_lines=[("n", n), ("rho", rho)],
        payload={
            "sigma": np.asarray(sig, dtype=float),
            "precision": np.asarray(prec, dtype=float),
            "logdet": logdet,
            "n": n,
            "rho": rho,
            "sigma2": sigma2,
        },
    )


statistical_methods_for_spatial_data_analysis_chapter_1_equation_28 = ar1cov


def cheatsheet():
    return "ar1cov: Sigma_ij = sigma2 rho^|i-j| with the tridiagonal closed-form inverse."
