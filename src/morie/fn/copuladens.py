"""Bivariate copula densities: Gaussian, Student t, Clayton, Gumbel, Frank, Joe.

Joe, H. (2014). Dependence Modeling with Copulas. CRC Press, ch. 4.
"""

import math

from ._richresult import RichResult
from ._rrng_core import qnorm, qt

__all__ = ["copuladens"]


def _dt(x, nu):
    return math.exp(
        math.lgamma((nu + 1) / 2)
        - math.lgamma(nu / 2)
        - 0.5 * math.log(nu * math.pi)
        - (nu + 1) / 2 * math.log1p(x * x / nu)
    )


def copuladens(u, v, family="gaussian", theta=0.5, df=4.0, delta=1.5):
    r"""Copula density c(u, v) = d^2 C(u, v)/(du dv).

    gaussian (theta = rho): exp(-(rho^2 (a^2 + b^2) - 2 rho a b)/(2(1 - rho^2)))/sqrt(1 - rho^2), a = Phi^{-1}(u);
    t (theta = rho, df): t_2 density at (a, b) over t_1(a) t_1(b), a = t_df^{-1}(u);
    clayton: (1 + theta)(u v)^{-theta-1}(u^{-theta} + v^{-theta} - 1)^{-2-1/theta};
    gumbel: C (x y)^{theta-1}/(u v) A^{1/theta-2} (A^{1/theta} + theta - 1), A = x^theta + y^theta, x = -log u;
    frank: theta(1 - e^{-theta}) e^{-theta(u+v)} / ((1 - e^{-theta}) - (1 - e^{-theta u})(1 - e^{-theta v}))^2;
    joe: (s)^{1/theta-2} ubar^{theta-1} vbar^{theta-1} (theta - 1 + s), s = ubar^theta + vbar^theta - ubar^theta vbar^theta;
    bb1 (theta > 0, delta >= 1): C = (1 + w)^{-1/theta}, w = s^{1/delta}, s = a^delta + b^delta, a = u^{-theta} - 1,
    differentiated in closed form; delta = 1 is Clayton, theta -> 0 is Gumbel(delta) (Joe 2014, sec 4.17).

    Parameters
    ----------
    u, v : float in (0, 1)
    family : {"gaussian", "t", "clayton", "gumbel", "frank", "joe", "bb1"}
    theta : float
    df : float
        Degrees of freedom of the t copula.
    delta : float
        Second BB1 parameter, delta >= 1.

    Returns
    -------
    RichResult
        Keys: density, logdensity.

    References
    ----------
    Joe, H. (2014). Dependence Modeling with Copulas, ch. 4.
    Matches ``copula::dCopula``.

    Examples
    --------
    >>> copuladens(0.3, 0.7, "clayton", 1e-12)["density"] > 0.999
    True
    """
    if not (0 < u < 1 and 0 < v < 1):
        raise ValueError("u and v must be in (0, 1)")
    if family == "gaussian":
        if not -1 < theta < 1:
            raise ValueError("rho must be in (-1, 1)")
        a, b = qnorm(u), qnorm(v)
        r = theta
        d = math.exp(-(r * r * (a * a + b * b) - 2 * r * a * b) / (2 * (1 - r * r))) / math.sqrt(1 - r * r)
    elif family == "t":
        if not (-1 < theta < 1 and df > 0):
            raise ValueError("rho must be in (-1, 1) and df > 0")
        a, b = qt(u, df), qt(v, df)
        r = theta
        q = (a * a - 2 * r * a * b + b * b) / (1 - r * r)
        l2 = (
            math.lgamma((df + 2) / 2)
            - math.lgamma(df / 2)
            - math.log(df * math.pi)
            - 0.5 * math.log(1 - r * r)
            - (df + 2) / 2 * math.log1p(q / df)
        )
        d = math.exp(l2) / (_dt(a, df) * _dt(b, df))
    elif family == "clayton":
        if not theta > 0:
            raise ValueError("theta must be > 0")
        d = (1 + theta) * (u * v) ** (-theta - 1) * (u**-theta + v**-theta - 1) ** (-2 - 1 / theta)
    elif family == "gumbel":
        if not theta >= 1:
            raise ValueError("theta must be >= 1")
        x, y = -math.log(u), -math.log(v)
        A = x**theta + y**theta
        C = math.exp(-(A ** (1 / theta)))
        d = C * (x * y) ** (theta - 1) / (u * v) * A ** (1 / theta - 2) * (A ** (1 / theta) + theta - 1)
    elif family == "frank":
        if theta == 0:
            raise ValueError("theta must be non-zero")
        e = -math.expm1(-theta)
        den = e - (-math.expm1(-theta * u)) * (-math.expm1(-theta * v))
        d = theta * e * math.exp(-theta * (u + v)) / (den * den)
    elif family == "joe":
        if not theta >= 1:
            raise ValueError("theta must be >= 1")
        ub, vb = 1 - u, 1 - v
        s = ub**theta + vb**theta - (ub * vb) ** theta
        d = s ** (1 / theta - 2) * ub ** (theta - 1) * vb ** (theta - 1) * (theta - 1 + s)
    elif family == "bb1":
        if not (theta > 0 and delta >= 1):
            raise ValueError("need theta > 0 and delta >= 1")
        a, b = u**-theta - 1, v**-theta - 1
        s = a**delta + b**delta
        w = s ** (1 / delta)
        xu = delta * a ** (delta - 1) * theta * u ** (-theta - 1)
        yv = delta * b ** (delta - 1) * theta * v ** (-theta - 1)
        h1 = -((1 + w) ** (-1 / theta - 1)) / theta
        h2 = (1 / theta) * (1 / theta + 1) * (1 + w) ** (-1 / theta - 2)
        d = xu * yv * (h2 * (w / (delta * s)) ** 2 + h1 * (1 / delta) * (1 / delta - 1) * w / (s * s))
    else:
        raise ValueError('family must be "gaussian", "t", "clayton", "gumbel", "frank", "joe" or "bb1"')
    return RichResult(
        title=f"{family} copula density",
        summary_lines=[("density", d)],
        payload={"density": d, "logdensity": math.log(d)},
    )


def cheatsheet():
    return "copuladens: bivariate copula densities (Gaussian, t, Clayton, Gumbel, Frank, Joe)."
