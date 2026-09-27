"""Von Mises-Fisher density on the unit sphere S^{p-1}.

Mardia, K. V. & Jupp, P. E. (2000). Directional Statistics. Wiley, Sec 9.3.2.
"""

import math

from ._mvcore import log_bessel_i, points
from ._richresult import RichResult

__all__ = ["vmfdens"]


def vmfdens(x, mu, kappa):
    r"""f(x) = C_p(kappa) exp(kappa mu' x), C_p = kappa^{p/2 - 1} / ((2 pi)^{p/2} I_{p/2 - 1}(kappa)).

    kappa = 0 is the uniform density 1/|S^{p-1}| = Gamma(p/2)/(2 pi^{p/2}).

    Parameters
    ----------
    x : unit vector or list of unit vectors
    mu : unit mean direction
    kappa : float
        Concentration, >= 0.

    Returns
    -------
    RichResult
        Keys: pdf, logpdf, log_normalizer.

    References
    ----------
    Mardia, K. V. & Jupp, P. E. (2000). Directional Statistics, Sec 9.3.2.

    Examples
    --------
    >>> round(vmfdens([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.0)["pdf"] * 4 * math.pi, 12)
    1.0
    """
    m = [float(v) for v in mu]
    p = len(m)
    if p < 2 or abs(sum(v * v for v in m) - 1) > 1e-9 or kappa < 0:
        raise ValueError("mu must be a unit vector in dimension >= 2 and kappa >= 0")
    if kappa == 0:
        logc = math.lgamma(p / 2) - math.log(2) - p / 2 * math.log(math.pi)
    else:
        logc = (p / 2 - 1) * math.log(kappa) - p / 2 * math.log(2 * math.pi) - log_bessel_i(p / 2 - 1, kappa)
    pts, single = points(x)
    lp = [logc + kappa * sum(a * b for a, b in zip(pt, m)) for pt in pts]
    return RichResult(
        title="Von Mises-Fisher",
        summary_lines=[("kappa", kappa)],
        payload={
            "pdf": math.exp(lp[0]) if single else [math.exp(v) for v in lp],
            "logpdf": lp[0] if single else lp,
            "log_normalizer": logc,
        },
    )


def cheatsheet():
    return "vmfdens: von Mises-Fisher density C_p(kappa) exp(kappa mu'x)."
