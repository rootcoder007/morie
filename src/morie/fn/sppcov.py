# morie.fn -- function file (rootcoder007/morie)
"""Spatial panel covariance structure."""

import math

from ._containers import SpatialResult
from ._qpcore import ssum


def sppcov(resid, unit_id, time_id):
    r"""Cross-sectional covariance of panel residuals and Pesaran's (2004) CD test.

    With residuals ``e_it`` the contemporaneous covariance is ``Sigma_ij =
    (1/T) sum_t e_it e_jt`` and the pairwise correlations ``rho_ij`` are
    computed over time; the cross-sectional dependence statistic ``CD =
    sqrt(2T / (N(N - 1))) sum_{i<j} rho_ij`` is asymptotically N(0, 1) under
    cross-sectional independence (two-sided p-value), as
    ``plm::pcdtest(test = "cd")``.

    References
    ----------
    Pesaran, M. H. (2004). General diagnostic tests for cross section
    dependence in panels. CESifo Working Paper 1229 / IZA DP 1240.

    Examples
    --------
    >>> e = [0.5, -0.2, 0.1, 0.3, -0.4, 0.2, -0.1, 0.6, 0.0, 0.25, -0.3, 0.1]
    >>> round(sppcov(e, [0, 1, 2] * 4, [t for t in range(4) for _ in range(3)]).statistic, 10)
    -1.2320322143
    """
    e = [float(v) for v in (resid.tolist() if hasattr(resid, "tolist") else resid)]
    u = list(unit_id.tolist() if hasattr(unit_id, "tolist") else unit_id)
    t = list(time_id.tolist() if hasattr(time_id, "tolist") else time_id)
    units, periods = sorted(set(u)), sorted(set(t))
    N, T = len(units), len(periods)
    pos = {(u[k], t[k]): e[k] for k in range(len(e))}
    E = [[pos[(i, s)] for s in periods] for i in units]
    cov = [[ssum(E[i][s] * E[j][s] for s in range(T)) / T for j in range(N)] for i in range(N)]
    mean = [ssum(r) / T for r in E]
    dev = [[v - m for v in r] for r, m in zip(E, mean)]
    ss = [math.sqrt(ssum(v * v for v in r)) for r in dev]
    cor = [[ssum(a * b for a, b in zip(dev[i], dev[j])) / (ss[i] * ss[j]) for j in range(N)] for i in range(N)]
    cd = math.sqrt(2.0 * T / (N * (N - 1.0))) * ssum(cor[i][j] for i in range(N) for j in range(i + 1, N))
    p = math.erfc(abs(cd) / math.sqrt(2.0))
    return SpatialResult(name="sppcov", statistic=cd, p_value=p, extra={"covariance": cov, "correlation": cor})


sppcov_fn = sppcov


def cheatsheet() -> str:
    return "sppcov(resid, unit_id, time_id) -> residual cross-sectional covariance and Pesaran CD test."
