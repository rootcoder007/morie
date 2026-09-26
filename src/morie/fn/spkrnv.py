"""Parametric kernel estimator of the semivariogram, Schabenberger & Gotway eq (4.50)."""

import math

from ._richresult import RichResult
from ._schab_npvg import kernel_correlation, nelder_mead

__all__ = ["kernel_semivariogram"]


def kernel_semivariogram(h, sill=None, theta_l=None, theta_u=None, d=2, b=1.0, gamma_hat=None, start=(0.0, 0.2)):
    r"""Semivariogram from a uniform spectral kernel, and its least-squares fit.

    Schabenberger & Gotway (2005, Sec. 4.6.1.2, eqs 4.48-4.51): with
    :math:`F(\theta, \omega)` the :math:`U(\theta_l, \theta_u)` cdf on
    :math:`0 \le \omega \le b`, zero below and one above (4.49),

    .. math::

        C(\theta, h) = \sigma^2\int_0^b \Omega_d(h\omega)\,F(\theta, d\omega),
        \qquad \gamma(h) = \sigma^2 - C(\theta, h).

    When :math:`\theta_l < 0` or :math:`\theta_u > b`, F has atoms at 0 and
    at b, and they are included; for :math:`0 \le \theta_l < \theta_u \le b`
    this is the density form (4.50). The integral is evaluated by
    composite Gauss-Legendre quadrature.

    Evaluate mode takes ``sill``, ``theta_l``, ``theta_u``. Fit mode takes
    ``gamma_hat`` at lags ``h`` and minimises the ordinary least squares
    criterion (4.51), :math:`Q(\theta) = \sum_k \{\hat\gamma(h_k) - (\sigma^2
    - C(\theta, h_k))\}^2`, with :math:`\sigma^2` profiled out in closed form
    and Nelder-Mead over :math:`(\sqrt{\theta_l}, \log(\theta_u - \theta_l))`
    from ``start``. The fit keeps :math:`\theta_l \ge 0`: for
    :math:`\theta_l < 0` the atom at zero adds a constant to the
    correlation, so only :math:`\theta_u` and
    :math:`\sigma^2\theta_u/(\theta_u - \theta_l)` are identified and the
    criterion is flat along a line.

    Parameters
    ----------
    h : sequence of float
        Lags, non-negative.
    sill, theta_l, theta_u : float, optional
        Model parameters (evaluate mode), ``theta_l < theta_u``.
    d : int
        Dimension, 1, 2 or 3.
    b : float
        Upper bound of the spectral support.
    gamma_hat : sequence of float, optional
        Empirical semivariogram at ``h`` (fit mode).
    start : tuple
        Starting ``(theta_l, theta_u - theta_l)`` for the fit.

    Returns
    -------
    RichResult
        ``gamma``, ``covariance``, ``sill``, ``theta_l``, ``theta_u``,
        ``sill_effective`` (:math:`\sigma^2\theta_u/(\theta_u - \theta_l)`
        when :math:`\theta_l \ge 0`, the combination the data pin down near
        :math:`\theta_l = 0`), and in fit mode ``q`` (the minimised
        criterion).

    References
    ----------
    Schabenberger, O. & Gotway, C. A. (2005). Statistical Methods for
    Spatial Data Analysis. Chapman & Hall/CRC, eqs (4.48)-(4.51),
    pp. 181-183.
    """
    h = [float(v) for v in h]
    b = float(b)
    q = None
    if gamma_hat is not None:
        g = [float(v) for v in gamma_hat]
        if len(g) != len(h):
            raise ValueError("`gamma_hat` must match `h`")

        def profile(p):
            k = kernel_correlation(h, p[0] ** 2, p[0] ** 2 + math.exp(p[1]), d, b)
            one = [0.0 if hv == 0 else 1.0 - kv for hv, kv in zip(h, k)]
            den = sum(v * v for v in one)
            s2 = sum(gv * v for gv, v in zip(g, one)) / den if den > 0 else 0.0
            return s2, sum((gv - s2 * v) ** 2 for gv, v in zip(g, one))

        if float(start[0]) < 0 or not float(start[1]) > 0:
            raise ValueError("`start` needs theta_l >= 0 and a positive width")
        best, q = nelder_mead(lambda p: profile(p)[1], [math.sqrt(float(start[0])), math.log(float(start[1]))])
        theta_l, theta_u = best[0] ** 2, best[0] ** 2 + math.exp(best[1])
        sill = profile(best)[0]
    if sill is None or theta_l is None or theta_u is None:
        raise ValueError("give `sill`, `theta_l`, `theta_u` to evaluate or `gamma_hat` to fit")
    sill, theta_l, theta_u = float(sill), float(theta_l), float(theta_u)
    if not theta_u > theta_l:
        raise ValueError("`theta_u` must exceed `theta_l`")
    k = kernel_correlation(h, theta_l, theta_u, d, b)
    cov = [sill * v for v in k]
    gam = [0.0 if hv == 0 else sill - c for hv, c in zip(h, cov)]
    payload = {
        "gamma": gam,
        "covariance": cov,
        "sill": sill,
        "theta_l": theta_l,
        "sill_effective": sill * theta_u / (theta_u - theta_l),
        "theta_u": theta_u,
        "d": d,
        "b": b,
    }
    if q is not None:
        payload["q"] = q
    return RichResult(
        title="Parametric kernel semivariogram (eq 4.50)",
        summary_lines=[("sill", sill), ("theta_l", theta_l), ("theta_u", theta_u)],
        payload=payload,
    )


def cheatsheet():
    return "spkrnv: gamma = sigma2 - sigma2 int Omega_d(h w) dF(theta, w), uniform kernel; OLS fit (4.51)"
