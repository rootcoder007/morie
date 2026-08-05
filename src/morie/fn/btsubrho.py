# morie.fn -- wave 2 slice x_0_01 (rootcoder007/morie)
"""Adaptive choice of the resample size m in the m-out-of-n bootstrap.

Bickel, P. J. and Sakov, A. (2008), "On the choice of m in the m out of
n bootstrap and confidence bounds for extrema", *Statistica Sinica*
18(3), 967-985.  The rule is stated verbatim on page 971 and was read
from the journal PDF:

  1. Consider a sequence of m's of the form m_j = ceil(q^j n),
     j = 0, 1, 2, ..., 0 < q < 1.
  2. For each m_j find the bootstrap law L*_{m_j, n}.
  3. With rho a metric consistent with convergence in law,
         m_hat = argmin_{m_j} rho( L*_{m_j,n}, L*_{m_{j+1},n} ),
     and "if the difference is minimized for a few values of m_j, then
     pick the LARGEST among them".
  4. Estimate L by L*_{m_hat, n}.

The paper's own choice of rho, and the one its proofs are for, is the
Kolmogorov sup distance sup_x |F(x) - G(x)|; that is what is used here.

The stub this replaces labelled the method "min-volatility", which is a
different rule (Politis, Romano and Wolf 1999, the standard deviation of
interval endpoints over a window of neighbouring m).  The citation on the
stub is Bickel and Sakov, so the Bickel-Sakov rule is what is
implemented; the ``vol_curve`` key is kept and carries the KS
discrepancies rho(L_j, L_{j+1}), which is the quantity the rule
minimises.

The law compared is that of the root sqrt(m)(theta*_m - theta_hat), which
is the normalisation under which the m-bootstrap law converges.  Ties are
broken toward the largest m exactly as the paper directs.

Anchor: on a constant sample every root is zero at every m, so every KS
discrepancy is zero, the argmin is a full tie, and the rule must return
the LARGEST grid value.  A rule that broke ties the other way would be
invisible on any non-degenerate fixture.

Resampling uses the package's shared Lehmer stream; the stream is
restarted at the same seed for every grid point so that the comparison
across m is not confounded by different noise.
"""

from __future__ import annotations

import math

from . import _s03core as core
from . import _tail1core as C

from ._richresult import RichResult
from .btmoutn import resample_idx

__all__ = ["boot_subsample_rate"]


def _ks(a, b):
    """Kolmogorov sup distance between two empirical cdfs."""
    pts = sorted(set(a) | set(b))
    na = float(len(a))
    nb = float(len(b))
    sa = sorted(a)
    sb = sorted(b)
    d = 0.0
    ia = 0
    ib = 0
    for t in pts:
        while ia < len(sa) and sa[ia] <= t:
            ia += 1
        while ib < len(sb) and sb[ib] <= t:
            ib += 1
        e = abs(ia / na - ib / nb)
        if e > d:
            d = e
    return d


def boot_subsample_rate(x, stat=None, m_grid=None, B=200, seed=1, q=0.75):
    """Pick m by the Bickel-Sakov adjacent-KS rule.

    Parameters
    ----------
    x : array-like
        The observed sample.
    stat : callable, optional
        Statistic of a sample.  Defaults to the mean.
    m_grid : sequence of int, optional
        The grid, largest first.  Defaults to ``ceil(q^j n)`` for
        j = 0, 1, ... down to the first value below 2.
    B : int
        Bootstrap replicates per grid point.
    seed : int
        Seed for the shared deterministic stream.
    q : float
        Grid ratio, ``0 < q < 1``.  Ignored when ``m_grid`` is given.

    Returns
    -------
    RichResult
        ``m_star``, ``vol_curve`` (the KS discrepancies, one shorter
        than the grid), ``m_grid``, ``theta_hat``, ``se_star``
        (rescaled replicate spread at ``m_star``), ``n``, ``B``.
    """
    xx = core.vec(x)
    n = len(xx)
    if n < 2:
        raise ValueError("boot_subsample_rate: need at least two observations")
    if int(B) < 2:
        raise ValueError("boot_subsample_rate: need at least two replicates")
    if m_grid is None:
        qq = float(q)
        if not (0.0 < qq < 1.0):
            raise ValueError("boot_subsample_rate: q must lie strictly between 0 and 1")
        m_grid = []
        j = 0
        pw = 1.0
        while True:
            mj = int(math.ceil(pw * n))
            if mj < 2:
                break
            if not m_grid or mj != m_grid[-1]:
                m_grid.append(mj)
            pw = pw * qq
            j += 1
            if j > 200:
                break
    m_grid = [int(u) for u in m_grid]
    if len(m_grid) < 2:
        raise ValueError("boot_subsample_rate: need at least two grid values")
    for mj in m_grid:
        if not 1 <= mj <= n:
            raise ValueError("boot_subsample_rate: every grid value must lie in 1..n")
    f = core.mean if stat is None else stat
    th = float(f(xx))
    laws = []
    for mj in m_grid:
        g = C.Lcg(seed)
        r = []
        for _ in range(int(B)):
            idx = resample_idx(g, n, mj)
            r.append(math.sqrt(mj) * (float(f([xx[j] for j in idx])) - th))
        laws.append(r)
    vol = [_ks(laws[i], laws[i + 1]) for i in range(len(m_grid) - 1)]
    best = min(vol)
    # ties -> the largest m, and m_grid is in decreasing order, so the
    # first index attaining the minimum is the one to take.
    k = 0
    for i in range(len(vol)):
        if vol[i] <= best:
            k = i
            break
    ms = m_grid[k]
    sdr = core.sd(laws[k], 1)
    return RichResult(
        title="Bickel-Sakov choice of m",
        summary_lines=[("n", n), ("m_star", ms), ("min_ks", best)],
        payload={
            "m_star": ms,
            "vol_curve": vol,
            "m_grid": m_grid,
            "min_ks": best,
            "theta_hat": th,
            "se_star": sdr / math.sqrt(n),
            "n": n,
            "B": int(B),
            "estimate": th,
            "method": "Bickel and Sakov (2008) Statist. Sinica 18(3):967-985, rule on p.971",
        },
    )


def cheatsheet():
    return "btsubrho: m_j = ceil(q^j n); minimise the KS gap between adjacent bootstrap laws; ties -> largest m"
