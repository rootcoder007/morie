"""Sample size and power for testing a correlation coefficient."""

import math

from ._richresult import RichResult
from ._stats_core import norm

__all__ = ["correlation_sample_size"]


def correlation_sample_size(r, alpha=0.05, power=None, n=None):
    r"""Sample size or power for the two-sided test of :math:`\rho = 0` at :math:`\rho = r`.

    Hedderich, Sachs & Reynarowych (2023, eqs 7.404-7.405): with
    :math:`\dot z_r = \tanh^{-1} r`,
    :math:`\dot z_r\sqrt{n-3} = z_{1-\alpha/2} + z_\beta`, so

    .. math::

        n = \left(\frac{z_{1-\alpha/2} + z_{1-\beta}}{\tanh^{-1} r}\right)^2 + 3,
        \qquad \mathrm{power} = \Phi\!\left(\sqrt{n-3}\,\tanh^{-1}r - z_{1-\alpha/2}\right).

    Give exactly one of ``power`` and ``n``. (pwr's ``pwr.r.test`` adds a
    bias correction to :math:`\dot z_r` and so asks for slightly fewer pairs.)

    Parameters
    ----------
    r : float
        Correlation to detect, non-zero.
    alpha : float
    power : float, optional
    n : int, optional

    Returns
    -------
    RichResult
        ``n`` (rounded up) and ``n_exact``, or ``power``.

    References
    ----------
    Hedderich, J., Sachs, L. & Reynarowych, Z. (2023). Applied Statistics:
    Methods Using R. Springer, eqs (7.404)-(7.405), Table 7.83.
    """
    r = float(r)
    if not -1 < r < 1 or r == 0:
        raise ValueError("`r` must be non-zero and in (-1, 1)")
    if (power is None) == (n is None):
        raise ValueError("give exactly one of `power` and `n`")
    za = float(norm.ppf(1 - alpha / 2))
    zr = abs(math.atanh(r))
    if n is None:
        ne = ((za + float(norm.ppf(power))) / zr) ** 2 + 3
        payload = {"n": math.ceil(ne), "n_exact": ne}
    else:
        payload = {"power": float(norm.cdf(math.sqrt(int(n) - 3) * zr - za)), "n": int(n)}
    payload.update({"r": r, "alpha": alpha})
    return RichResult(title="Correlation sample size / power", summary_lines=list(payload.items())[:2], payload=payload)


def cheatsheet():
    return "corss: n = ((z_{1-a/2} + z_{1-b}) / atanh r)^2 + 3 for the test of rho = 0"
