# morie.fn -- wave2 slice w2_02 (rootcoder007/morie)
"""Generalizability theory: the crossed person-by-item G study.

Cronbach, Gleser, Nanda and Rajaratnam (1972), *The Dependability of
Behavioral Measurements*, Wiley; see also Brennan (2001),
*Generalizability Theory*, Springer, chapter 2.  For the p x i design
the random-effects ANOVA gives

    sigma^2_p  = (MS_p  - MS_pi) / n_i,
    sigma^2_i  = (MS_i  - MS_pi) / n_p,
    sigma^2_pi = MS_pi,

and the generalizability coefficient (relative decisions) and index of
dependability (absolute decisions) are

    E rho^2 = sigma^2_p / (sigma^2_p + sigma^2_pi / n_i),
    Phi     = sigma^2_p / (sigma^2_p + (sigma^2_i + sigma^2_pi) / n_i).

For this design E rho^2 is algebraically identical to coefficient
alpha, which is the independent check used in the tests.
"""

from __future__ import annotations

from . import _array_core as np  # noqa: F401
from . import _s03core as core

from ._richresult import RichResult

__all__ = ["generalizability_theory"]


def generalizability_theory(X, facets=None):
    """G study of a persons-by-items score matrix.

    Parameters
    ----------
    X : n_p x n_i matrix of scores.
    facets : int, optional
        Number of items to project onto; defaults to the observed n_i.
    """
    M = core.mat(X)
    npr = len(M)
    if npr < 2:
        raise ValueError("generalizability_theory: need at least two persons")
    ni = len(M[0])
    if ni < 2:
        raise ValueError("generalizability_theory: need at least two items")
    for r in M:
        if len(r) != ni:
            raise ValueError("generalizability_theory: X must be rectangular")
    grand = sum(sum(r) for r in M) / (npr * ni)
    pm = [sum(M[i]) / ni for i in range(npr)]
    im = [sum(M[i][j] for i in range(npr)) / npr for j in range(ni)]
    msp = ni * sum((v - grand) ** 2 for v in pm) / (npr - 1.0)
    msi = npr * sum((v - grand) ** 2 for v in im) / (ni - 1.0)
    res = 0.0
    for i in range(npr):
        for j in range(ni):
            res += (M[i][j] - pm[i] - im[j] + grand) ** 2
    mspi = res / ((npr - 1.0) * (ni - 1.0))
    vp = (msp - mspi) / ni
    vi = (msi - mspi) / npr
    vpi = mspi
    k = ni if facets is None else int(facets)
    if k < 1:
        raise ValueError("generalizability_theory: facets must be at least 1")
    erho = vp / (vp + vpi / k) if (vp + vpi / k) != 0 else float("nan")
    phi = vp / (vp + (vi + vpi) / k) if (vp + (vi + vpi) / k) != 0 else float("nan")
    return RichResult(
        title="G study, p x i crossed design",
        summary_lines=[("persons", npr), ("items", ni)],
        payload={
            "estimate": erho,
            "e_rho2": erho,
            "phi": phi,
            "var_p": vp,
            "var_i": vi,
            "var_pi": vpi,
            "ms_p": msp,
            "ms_i": msi,
            "ms_pi": mspi,
            "n_p": npr,
            "n_i": k,
            "method": "random-effects ANOVA components, Cronbach et al. (1972); Brennan (2001) ch. 2",
        },
    )


def cheatsheet():
    return "genvxt: generalizability coefficient (G study)"
