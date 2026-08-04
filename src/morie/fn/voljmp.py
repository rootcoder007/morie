"""Barndorff-Nielsen & Shephard jump test from bipower variation."""

from math import gamma as _gamma, pi, sqrt

from . import _array_core as np
from . import _stats_core as stats
from ._richresult import hypothesis_test_result

__all__ = ["vol_jump_test_bnshep"]

# theta for the bipower IV estimator, Barndorff-Nielsen & Shephard (2006):
# theta = pi^2/4 + pi - 3, and the test's denominator carries theta - 2.
_THETA_BV = pi * pi / 4.0 + pi - 3.0
# mu_{4/3}^{-3} for tripower quarticity: mu_{4/3} = 2^{2/3} Gamma(7/6)/Gamma(1/2)
_MU43_INV3 = (_gamma(0.5) / (2.0 ** (2.0 / 3.0) * _gamma(7.0 / 6.0))) ** 3


def _rv(r):
    return float(sum(v * v for v in r))


def _bpv(r):
    a = [abs(v) for v in r]
    return float((pi / 2.0) * sum(a[i] * a[i - 1] for i in range(1, len(a))))


def _tpq(r):
    """Realised tripower quarticity, the jump-robust IQ estimator."""
    a = [abs(v) for v in r]
    n = len(a)
    s = sum((a[i] * a[i - 1] * a[i - 2]) ** (4.0 / 3.0) for i in range(2, n))
    return float(n * (n / (n - 2.0)) * _MU43_INV3 * s)


def _one_day(r):
    n = len(r)
    rv, bpv, tpq = _rv(r), _bpv(r), _tpq(r)
    denom = sqrt((_THETA_BV - 2.0) * tpq / n)
    z = (rv - bpv) / denom if denom > 0 else float("nan")
    return rv, bpv, tpq, z


def vol_jump_test_bnshep(r_intraday, block_index=None):
    r"""Barndorff-Nielsen & Shephard (2006) test for jumps.

    Realised variance converges to integrated variance *plus* jump
    variation; bipower variation converges to integrated variance
    alone, because a jump enters each product beside an
    :math:`O(\sqrt{dt})` diffusive neighbour. Their difference is
    therefore a jump detector, and the linear form of the statistic is

    .. math::

       z = \frac{RV - BPV}{\sqrt{(\theta - 2)\,N^{-1} TP}},
       \qquad \theta = \tfrac{\pi^2}{4} + \pi - 3,

    asymptotically standard normal under "no jumps". ``TP`` is realised
    tripower quarticity, itself jump-robust -- using plain realised
    quarticity in the denominator would let the very jump being tested
    for inflate the standard error and destroy the test's power.

    The test is one-sided: only a *positive* z is evidence of jumps,
    since RV can only exceed BPV under the alternative.

    Parameters
    ----------
    r_intraday : array-like, shape (m,)
        Intraday log-returns. At least 4 are needed for tripower
        quarticity to be defined.
    block_index : array-like, optional
        Day (or block) label per return. Statistics are computed within
        block and never straddle a boundary; the overnight return is
        not a diffusive increment.

    Returns
    -------
    RichResult
        Keys ``statistic`` (z), ``pvalue`` (one-sided upper tail),
        ``rv``, ``bpv``, ``tpq``, ``jump_component`` = RV - BPV, and
        ``days`` when ``block_index`` is given.

    References
    ----------
    Barndorff-Nielsen, O. E. & Shephard, N. (2006). Econometrics of
    testing for jumps in financial economics using bipower variation.
    *Journal of Financial Econometrics*, 4(1), 1-30.
    Definition cross-checked against the reference implementation in
    the highfrequency R package (``BNSjumpTest``, ``tt``, ``rTPQuar``).
    """
    r = [float(v) for v in np.asarray(r_intraday, dtype=float).ravel().tolist()]
    if len(r) < 4:
        raise ValueError("need at least 4 intraday returns for tripower quarticity.")
    if block_index is None:
        rv, bpv, tpq, z = _one_day(r)
        return hypothesis_test_result(
            test_name="Barndorff-Nielsen-Shephard jump test",
            statistic=float(z),
            pvalue=float(stats.norm.sf(z)),
            extra_summary=[("n_returns", len(r)), ("rv", rv), ("bpv", bpv)],
            extra_payload={
                "rv": rv, "bpv": bpv, "tpq": tpq,
                "jump_component": rv - bpv,
                "n_returns": len(r), "days": None,
                "method": "BNS (2006) linear jump test, bipower vs realised variance",
            },
        )
    d = [x for x in np.asarray(block_index).ravel().tolist()]
    if len(d) != len(r):
        raise ValueError("block_index must have one entry per return.")
    days = list(dict.fromkeys(d))
    out = {}
    for day in days:
        v = [r[i] for i in range(len(r)) if d[i] == day]
        if len(v) < 4:
            raise ValueError("each block needs at least 4 returns.")
        out[day] = _one_day(v)
    zs = [out[day][3] for day in days]
    return hypothesis_test_result(
        test_name="Barndorff-Nielsen-Shephard jump test (per block)",
        statistic=float(max(zs)),
        pvalue=float(stats.norm.sf(max(zs))),
        extra_summary=[("n_blocks", len(days)), ("n_returns", len(r))],
        extra_payload={
            "days": days,
            "z": zs,
            "pvalue_by_block": [float(stats.norm.sf(v)) for v in zs],
            "rv": [out[day][0] for day in days],
            "bpv": [out[day][1] for day in days],
            "tpq": [out[day][2] for day in days],
            "jump_component": [out[day][0] - out[day][1] for day in days],
            "n_returns": len(r),
            "method": "BNS (2006) linear jump test per block; statistic is the max z",
        },
    )


def cheatsheet():
    return "voljmp: Barndorff-Nielsen-Shephard jump test (RV vs bipower variation)"
