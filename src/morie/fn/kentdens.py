"""Kent (Fisher-Bingham five-parameter) density on the sphere S^2.

Kent, J. T. (1982). The Fisher-Bingham distribution on the sphere. JRSS B 44, 71-80.
"""

import math

from ._mvcore import log_bessel_i, points
from ._richresult import RichResult

__all__ = ["kentdens"]


def kentdens(x, kappa, beta, g1=(0.0, 0.0, 1.0), g2=(1.0, 0.0, 0.0), g3=(0.0, 1.0, 0.0)):
    r"""f(x) = exp(kappa g1'x + beta ((g2'x)^2 - (g3'x)^2)) / c(kappa, beta), 0 <= 2 beta < kappa,

    c(kappa, beta) = 2 pi sum_{j>=0} Gamma(j + 1/2)/Gamma(j + 1) beta^{2j} (kappa/2)^{-2j-1/2} I_{2j+1/2}(kappa),
    summed until the terms fall below 1e-17 of the total; g1, g2, g3 are
    orthonormal (mean direction, major and minor axes).

    Parameters
    ----------
    x : unit 3-vector or list of them
    kappa, beta : float
    g1, g2, g3 : orthonormal unit vectors

    Returns
    -------
    RichResult
        Keys: pdf, logpdf, log_normalizer.

    References
    ----------
    Kent, J. T. (1982). JRSS B 44, 71-80, eq (2.2).

    Examples
    --------
    >>> r = kentdens([0.0, 0.0, 1.0], 2.0, 0.0)
    >>> round(r["pdf"], 12) == round(2.0 * math.exp(2.0) / (4 * math.pi * math.sinh(2.0)), 12)
    True
    """
    if not (kappa > 0 and 0 <= 2 * beta < kappa):
        raise ValueError("need kappa > 0 and 0 <= 2 beta < kappa")
    G = [[float(v) for v in g] for g in (g1, g2, g3)]
    for i in range(3):
        for j in range(3):
            if abs(sum(a * b for a, b in zip(G[i], G[j])) - (i == j)) > 1e-9:
                raise ValueError("g1, g2, g3 must be orthonormal")
    terms = []
    j = 0
    while True:
        t = (
            math.lgamma(j + 0.5)
            - math.lgamma(j + 1)
            + (2 * j * math.log(beta) if beta > 0 else (0.0 if j == 0 else -math.inf))
            - (2 * j + 0.5) * math.log(kappa / 2)
            + log_bessel_i(2 * j + 0.5, kappa)
        )
        terms.append(t)
        if j > 3 and (t < max(terms) - 40 or beta == 0):
            break
        j += 1
    top = max(terms)
    logc = math.log(2 * math.pi) + top + math.log(sum(math.exp(t - top) for t in terms))
    pts, single = points(x)
    lp = []
    for pt in pts:
        a, b, c = (sum(g * v for g, v in zip(G[i], pt)) for i in range(3))
        lp.append(kappa * a + beta * (b * b - c * c) - logc)
    return RichResult(
        title="Kent distribution",
        summary_lines=[("kappa", kappa), ("beta", beta)],
        payload={
            "pdf": math.exp(lp[0]) if single else [math.exp(v) for v in lp],
            "logpdf": lp[0] if single else lp,
            "log_normalizer": logc,
        },
    )


def cheatsheet():
    return "kentdens: Kent FB5 density with series normalizing constant."
