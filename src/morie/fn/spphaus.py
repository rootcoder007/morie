# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel Hausman test (FE vs RE)."""

from ._containers import SpatialResult
from ._qpcore import inverse, ssum
from ._rrng_core import pchisq


def spphaus(coef_fe, coef_re, vcov_fe, vcov_re):
    r"""Hausman test of random against fixed effects: ``H = d'(V_FE - V_RE)^{-1} d``, ``d = b_FE - b_RE``.

    Under the null that the random effects are uncorrelated with the
    regressors both estimators are consistent and RE is efficient, so ``H``
    is asymptotically chi-square with ``k = len(d)`` df (Hausman 1978); for
    spatial panels the same contrast of the FE and RE spatial estimates is
    ``splm::sphtest`` (Mutl and Pfaffermayr 2011).

    References
    ----------
    Hausman, J. A. (1978). Specification tests in econometrics.
    *Econometrica* 46, 1251-1271.
    Mutl, J. and Pfaffermayr, M. (2011). The Hausman test in a Cliff and Ord
    panel model. *Econometrics Journal* 14, 48-76.

    Examples
    --------
    >>> round(spphaus([1.0, 0.5], [0.8, 0.6], [[0.05, 0.01], [0.01, 0.04]], [[0.02, 0.0], [0.0, 0.03]]).statistic, 10)
    5.5
    """
    d = [float(a) - float(b) for a, b in zip(coef_fe, coef_re)]
    k = len(d)
    V = [[float(vcov_fe[i][j]) - float(vcov_re[i][j]) for j in range(k)] for i in range(k)]
    Vi = inverse(V)
    h = ssum(d[i] * ssum(Vi[i][j] * d[j] for j in range(k)) for i in range(k))
    return SpatialResult(name="spphaus", statistic=h, p_value=float(pchisq(h, k, lower_tail=False)), extra={"df": k})


spphaus_fn = spphaus


def cheatsheet() -> str:
    return "spphaus(coef_fe, coef_re, vcov_fe, vcov_re) -> Hausman d'(V_FE - V_RE)^-1 d, chi2(k)."
