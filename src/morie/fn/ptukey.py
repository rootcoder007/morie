"""Distribution function of the studentized range (port of R's nmath ptukey).

Copenhaver & Holland (1988), Algorithm AS 190 lineage; used by Tukey-Kramer intervals,
Wilcox (2017), Modern Statistics for the Social and Behavioral Sciences, eq (12.4).
"""

import math

from ._richresult import RichResult
from ._rrng_core import pnorm

__all__ = ["ptukey"]

_XLEG = (
    0.981560634246719250690549090149,
    0.904117256370474856678465866119,
    0.769902674194304687036893833213,
    0.587317954286617447296702418941,
    0.367831498998180193752691536644,
    0.125233408511468915472441369464,
)
_ALEG = (
    0.047175336386511827194615961485,
    0.106939325995318430960254718194,
    0.160078328543346226334652529543,
    0.203167426723065921749064455810,
    0.233492536538354808760849898925,
    0.249147045813402785000562436043,
)
_XLEGQ = (
    0.989400934991649932596154173450,
    0.944575023073232576077988415535,
    0.865631202387831743880467897712,
    0.755404408355003033895101194847,
    0.617876244402643748446671764049,
    0.458016777657227386342419442984,
    0.281603550779258913230460501460,
    0.950125098376374401853193354250e-1,
)
_ALEGQ = (
    0.271524594117540948517805724560e-1,
    0.622535239386478928628438369944e-1,
    0.951585116824927848099251076022e-1,
    0.124628971255533872052476282192,
    0.149595988816576732081501730547,
    0.169156519395002538189312079030,
    0.182603415044923588866763667969,
    0.189450610455068496285396723208,
)


def _wprob(w, rr, cc):
    """P(range of cc standard normals < w), raised to rr (Hartley's form, Gauss-Legendre)."""
    qsqz = w * 0.5
    if qsqz >= 8.0:
        return 1.0
    pr_w = 2 * pnorm(qsqz) - 1.0
    pr_w = pr_w**cc if pr_w >= math.exp(-50.0 / cc) else 0.0
    wincr = 2.0 if w > 3.0 else 3.0
    blb = qsqz
    binc = (8.0 - qsqz) / wincr
    bub = blb + binc
    einsum = 0.0
    cc1 = cc - 1.0
    for _ in range(int(wincr)):
        elsum = 0.0
        a = 0.5 * (bub + blb)
        b = 0.5 * (bub - blb)
        for jj in range(1, 13):
            if jj > 6:
                j = 12 - jj + 1
                xx = _XLEG[j - 1]
            else:
                j = jj
                xx = -_XLEG[j - 1]
            ac = a + b * xx
            qexpo = ac * ac
            if qexpo > 60.0:
                break
            rinsum = pnorm(ac) - pnorm(ac, w, 1.0)
            if rinsum >= math.exp(-30.0 / cc1):
                elsum += _ALEG[j - 1] * math.exp(-0.5 * qexpo) * rinsum**cc1
        einsum += elsum * (2.0 * b * cc) / math.sqrt(2 * math.pi)
        blb = bub
        bub += binc
    pr_w += einsum
    if pr_w <= math.exp(-30.0 / rr):
        return 0.0
    pr_w = pr_w**rr
    return 1.0 if pr_w >= 1.0 else pr_w


def _ptukey(q, nranges, nmeans, df):
    if q <= 0:
        return 0.0
    if df < 2 or nranges < 1 or nmeans < 2:
        raise ValueError("need df >= 2, nranges >= 1 and nmeans >= 2")
    if math.isinf(q):
        return 1.0
    if df > 25000.0:
        return _wprob(q, nranges, nmeans)
    f2 = df * 0.5
    f2lf = f2 * math.log(df) - df * math.log(2) - math.lgamma(f2)
    f21 = f2 - 1.0
    ff4 = df * 0.25
    ulen = 1.0 if df <= 100 else 0.5 if df <= 800 else 0.25 if df <= 5000 else 0.125
    f2lf += math.log(ulen)
    ans = 0.0
    for i in range(1, 51):
        otsum = 0.0
        twa1 = (2 * i - 1) * ulen
        for jj in range(1, 17):
            if jj > 8:
                j = jj - 8 - 1
                t1 = f2lf + f21 * math.log(twa1 + _XLEGQ[j] * ulen) - (_XLEGQ[j] * ulen + twa1) * ff4
            else:
                j = jj - 1
                t1 = f2lf + f21 * math.log(twa1 - _XLEGQ[j] * ulen) + (_XLEGQ[j] * ulen - twa1) * ff4
            if t1 >= -30.0:
                if jj > 8:
                    qsqz = q * math.sqrt((_XLEGQ[j] * ulen + twa1) * 0.5)
                else:
                    qsqz = q * math.sqrt((-(_XLEGQ[j] * ulen) + twa1) * 0.5)
                otsum += _wprob(qsqz, nranges, nmeans) * _ALEGQ[j] * math.exp(t1)
        if i * ulen >= 1.0 and otsum <= 1.0e-14:
            break
        ans += otsum
    return min(ans, 1.0)


def ptukey(q, nmeans, df, nranges=1, lower_tail=True):
    """P(studentized range of ``nmeans`` means on ``df`` degrees of freedom <= q).

    A line-by-line port of R's ``ptukey`` (Copenhaver & Holland 1988): Hartley's
    form of the range distribution integrated by 12-point Gauss-Legendre
    quadrature, then integrated over the chi distribution of the scale with
    16-point quadrature on unit (or finer) intervals.

    Parameters
    ----------
    q : float
    nmeans : int
        Number of means (>= 2).
    df : float
        Degrees of freedom (>= 2).
    nranges : int
        Number of independent groups of ranges (1 for Tukey intervals).
    lower_tail : bool

    Returns
    -------
    RichResult
        Keys: p.

    References
    ----------
    Copenhaver, M. D. & Holland, B. S. (1988). Journal of Statistical
    Computation and Simulation 30, 1-15.

    Examples
    --------
    >>> round(ptukey(3.5, 3, 20)["p"], 7)
    0.9441082
    """
    p = _ptukey(float(q), float(nranges), float(nmeans), float(df))
    p = p if lower_tail else 1.0 - p
    return RichResult(title="Studentized range distribution", summary_lines=[("p", p)], payload={"p": p})


def cheatsheet():
    return "ptukey: studentized range cdf, port of R's ptukey (Copenhaver & Holland 1988)."
