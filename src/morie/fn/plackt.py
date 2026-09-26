"""Plackett's family of bivariate distributions."""

import math

from ._richresult import RichResult

__all__ = ["plackett_distribution"]


def plackett_distribution(u, v, psi):
    r"""Plackett's bivariate distribution function and its correlation.

    For marginal distribution values :math:`F_1 = u` and :math:`F_2 = v`,
    Plackett's (1965) :math:`F_{12}` is the root in :math:`[0, 1]` of

    .. math::

        \psi = \frac{F_{12}(1 - F_1 - F_2 + F_{12})}
                    {(F_1 - F_{12})(F_2 - F_{12})},\qquad \psi > 0,

    (Schabenberger & Gotway 2005, p. 293, print the denominator as
    :math:`(F_1 - F_{12})^2`; Plackett's is the product above), namely

    .. math::

        F_{12} = \frac{S - \sqrt{S^2 - 4\psi(\psi-1)uv}}{2(\psi-1)},\quad
        S = 1 + (\psi-1)(u+v),

    and :math:`F_{12} = uv` at :math:`\psi = 1` (independence). Mardia's
    (1967) correlation of the uniform margins (Spearman's rho) is

    .. math::

        \rho = \frac{(\psi-1)(1+\psi) - 2\psi\log\psi}{(1-\psi)^2}.

    Parameters
    ----------
    u, v : float or sequence of float
        Marginal distribution values in [0, 1], the same length.
    psi : float
        Association parameter, positive.

    Returns
    -------
    RichResult
        ``F12`` (list), ``rho`` (Mardia's correlation), ``psi``.

    References
    ----------
    Plackett, R. L. (1965). A class of bivariate distributions. JASA 60,
    516-522. Mardia, K. V. (1967). Some contributions to contingency-type
    bivariate distributions. Biometrika 54, 235-249. Schabenberger, O. &
    Gotway, C. A. (2005). Statistical Methods for Spatial Data Analysis.
    Chapman & Hall/CRC, p. 293.
    """
    psi = float(psi)
    if not psi > 0:
        raise ValueError("`psi` must be positive")
    a = [float(u)] if isinstance(u, (int, float)) else [float(x) for x in u]
    b = [float(v)] if isinstance(v, (int, float)) else [float(x) for x in v]
    if len(a) != len(b) or any(not 0 <= x <= 1 for x in a + b):
        raise ValueError("`u` and `v` must be the same length with values in [0, 1]")
    out = []
    for x, y in zip(a, b):
        if psi == 1.0:
            out.append(x * y)
        else:
            s = 1.0 + (psi - 1.0) * (x + y)
            out.append((s - math.sqrt(s * s - 4.0 * psi * (psi - 1.0) * x * y)) / (2.0 * (psi - 1.0)))
    rho = 0.0 if psi == 1.0 else ((psi - 1.0) * (1.0 + psi) - 2.0 * psi * math.log(psi)) / (1.0 - psi) ** 2
    return RichResult(
        title="Plackett bivariate distribution",
        summary_lines=[("psi", psi), ("rho", rho)],
        payload={"F12": out, "rho": rho, "psi": psi},
    )


def cheatsheet():
    return "plackt: Plackett F12(u, v; psi) and Mardia's correlation"
