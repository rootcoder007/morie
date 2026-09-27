"""Conditional distribution of one component of a bivariate normal given the other."""

import math

from ._richresult import RichResult

__all__ = ["bivariate_normal_conditional"]


def bivariate_normal_conditional(value, mu_x, mu_y, sd_x, sd_y, rho, given="y"):
    r"""Mean and standard deviation of X | Y = value (or Y | X = value).

    Hedderich, Sachs & Reynarowych (2023, p. 324):
    :math:`X \mid Y=y \sim N(\mu_x + \varrho\sigma_x(y-\mu_y)/\sigma_y,\ \sigma_x\sqrt{1-\varrho^2})`
    and symmetrically for :math:`Y \mid X = x`.

    Parameters
    ----------
    value : float
        Observed value of the conditioning variable.
    mu_x, mu_y : float
    sd_x, sd_y : float
        Standard deviations (> 0).
    rho : float
        Correlation in [-1, 1].
    given : {"y", "x"}
        Which variable is observed.

    Returns
    -------
    RichResult
        ``mean``, ``sd``.

    References
    ----------
    Hedderich, J., Sachs, L. & Reynarowych, Z. (2023). Applied Statistics:
    Methods Using R. Springer, Sec. 5.4.
    """
    if sd_x <= 0 or sd_y <= 0 or not -1 <= rho <= 1:
        raise ValueError("need positive standard deviations and -1 <= rho <= 1")
    if given == "y":
        m, s = mu_x + rho * sd_x * (value - mu_y) / sd_y, sd_x * math.sqrt(1 - rho * rho)
    elif given == "x":
        m, s = mu_y + rho * sd_y * (value - mu_x) / sd_x, sd_y * math.sqrt(1 - rho * rho)
    else:
        raise ValueError("`given` must be 'y' or 'x'")
    return RichResult(
        title=f"Conditional normal given {given}",
        summary_lines=[("mean", m), ("sd", s)],
        payload={"mean": m, "sd": s},
    )


def cheatsheet():
    return "bvncnd: mu_x + rho sd_x (y - mu_y)/sd_y, sd_x sqrt(1 - rho^2)"
