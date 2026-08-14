# morie.fn -- function file (rootcoder007/morie)
r"""The Christiano-Fitzgerald band pass filter.

The object wanted is the component of a series whose power lies in one
frequency band -- for quarterly macroeconomic data, periods of
oscillation between :math:`p_l` and :math:`p_u` quarters. Write the
orthogonal decomposition :math:`x_t = y_t + \tilde x_t` where
:math:`y_t` has power only in :math:`(a,b)\cup(-b,-a)`. Then
:math:`y_t = B(L)x_t` with

.. math:: B_j = \frac{\sin(jb) - \sin(ja)}{\pi j}, \quad j \ge 1,
          \qquad B_0 = \frac{b-a}{\pi},
          \qquad a = \frac{2\pi}{p_u},\ b = \frac{2\pi}{p_l}.

**The ideal filter is infinite, and truncating it is not innocuous.**
The paper's Figure 1a makes the point with monthly business-cycle
frequencies (:math:`a = 2\pi/96`, :math:`b = 2\pi/18`): the weights die
out only slowly, and even at :math:`j = 120` -- ten years of lags --
they remain noticeably different from zero. Truncation therefore has a
substantial effect on the filter's frequency response.

**So the finite-sample problem is a projection, not a truncation.**
With observations :math:`x = [x_1,\dots,x_T]`, the estimate is
:math:`\hat y = P[y \mid x]`, solved separately for each :math:`t`.
The solution depends on the time-series representation of the data --
unlike the ideal filter, which does not.

**The recommended filter assumes a random walk.** That assumption is,
in the authors' own words, most likely false, and it still works well
for US interest rates, unemployment, inflation and output. It gives
(eq. 1.2), for :math:`t = 3,\dots,T-2`,

.. math:: \hat y_t = B_0 x_t + B_1 x_{t+1} + \dots
          + B_{T-t-1}x_{T-1} + \tilde B_{T-t}x_T
          + B_1 x_{t-1} + \dots + B_{t-2}x_2 + \tilde B_{t-1}x_1,

where the endpoint weights absorb the truncated tails:
:math:`\tilde B_{T-t} = -\tfrac12 B_0 - \sum_{j=1}^{T-t-1}B_j`, using
:math:`B_0 + 2\sum_{j\ge1}B_j = 0`, and :math:`\tilde B_{t-1}` is fixed
by requiring the whole weight vector to sum to zero. That
zero-sum property is :math:`B(1) = 0` -- the filter annihilates a
constant, exactly -- and it is the first thing the anchor checks.

**Three routes, all implemented.**

* ``asymmetric`` (default) -- eq. (1.2). Uses every observation, so it
  produces an estimate at every date, but the weights vary with
  :math:`t` and the filter is not symmetric in leads and lags.
* ``symmetric`` -- a fixed number :math:`p` of leads and lags, with the
  endpoint weight :math:`B_p = -\tfrac12 B_0 - \sum_{j=1}^{p-1}B_j`
  (footnote 5). Symmetry buys the absence of phase shift, and costs the
  first and last :math:`p` observations.
* ``one_sided`` -- eq. (1.4), :math:`\hat y_T` from current and past
  data only. Needed when the estimate is required in real time, as in
  a stabilisation context.

**Drift matters, and only for one of them.** The symmetric
random-walk filter has two unit roots -- one makes the series
stationary, the other kills the drift -- so its output is invariant to
drift and no adjustment is needed. The asymmetric filter has only one,
so it is *not* invariant, and the raw data must be drift-adjusted
first with :math:`\hat\mu = (x_T - x_1)/(T-1)`. The anchor
demonstrates that asymmetry rather than assuming it.

**The caveat the authors insist on.** The recommended filter does not
approximate the optimal filter in all circumstances -- they show a case
where the first difference has substantial negative autocorrelation and
it works badly. Where that is a concern, the time-series
representation should be estimated and the optimal filter computed from
it.

References
----------
Christiano, L. J. & Fitzgerald, T. J. (2003) "The Band Pass Filter",
*International Economic Review* 44(2), 435-465,
doi:10.1111/1468-2354.t01-1-00076. Read here in the NBER working
paper version, "The Band Pass Filter", NBER Working Paper 7257 (1999),
doi:10.3386/w7257. Sec. 1: the recommended random-walk filter of
eqs. (1.2)-(1.3), the endpoint weights of footnote 3, the one-sided
filter of eq. (1.4), the fixed-lead-lag symmetric variant of footnote
5 and its invariance to drift, and the caveat about negatively
autocorrelated first differences. Sec. 2.1: the orthogonal
decomposition, :math:`B(e^{-i\omega}) = 1` on the band and 0 off it,
:math:`B(1) = 0`, and the Figure 1a observation that the weights
remain noticeably non-zero even at :math:`j = 120`. Sec. 2.2: the
finite-sample problem as the projection :math:`P[y\mid x]`.

Baxter, M. & King, R. G. (1999) "Measuring Business Cycles:
Approximate Band-Pass Filters for Economic Time Series", *Review of
Economics and Statistics* 81(4), 575-593,
doi:10.1162/003465399558454. The symmetric truncated filter this
paper is written against.

Hodrick, R. J. & Prescott, E. C. (1997) "Postwar U.S. Business Cycles:
An Empirical Investigation", *Journal of Money, Credit and Banking*
29(1), 1-16, doi:10.2307/2953682.
"""

import math

from . import _array_core as np
from . import _s03core as k
from ._richresult import RichResult

__all__ = ["ideal_weights", "cf_filter", "frequency_response",
           "drift_adjust"]

_EPS = 1e-12
_METHODS = ("asymmetric", "symmetric", "one_sided")


def ideal_weights(p_low, p_high, n):
    r"""Eq. (1.3): the ideal band pass weights :math:`B_0,\dots,B_n`.

    :math:`B_0 = (b-a)/\pi` and
    :math:`B_j = (\sin jb - \sin ja)/(\pi j)`, with
    :math:`a = 2\pi/p_u` and :math:`b = 2\pi/p_l`.
    """
    pl, pu = float(p_low), float(p_high)
    if not 2.0 <= pl < pu:
        raise ValueError("crfflt: need 2 <= p_low < p_high, got "
                         "(%.4f, %.4f)" % (pl, pu))
    if int(n) < 0:
        raise ValueError("crfflt: n must be non-negative")
    a, b = 2.0 * math.pi / pu, 2.0 * math.pi / pl
    B = [(b - a) / math.pi]
    for j in range(1, int(n) + 1):
        B.append((math.sin(j * b) - math.sin(j * a)) / (math.pi * j))
    return {"B": B, "a": a, "b": b, "p_low": pl, "p_high": pu,
            "B0": B[0],
            "note": "the ideal filter is infinite; these weights die "
                    "out only slowly (Fig. 1a: still non-zero at "
                    "j = 120)"}


def drift_adjust(x):
    r"""Remove the random walk's drift, :math:`\hat\mu=(x_T-x_1)/(T-1)`.

    Needed for the asymmetric filter, which has one unit root and is
    therefore not drift-invariant; the symmetric one has two and does
    not need it.
    """
    v = [float(q) for q in k.vec(x)]
    T = len(v)
    if T < 2:
        raise ValueError("crfflt: need at least 2 observations")
    mu = (v[-1] - v[0]) / (T - 1)
    return {"adjusted": [v[t] - t * mu for t in range(T)], "drift": mu}


def _tail(B, m):
    """:math:`\\tilde B_m = -\\tfrac12 B_0 - \\sum_{j=1}^{m-1}B_j`."""
    return -0.5 * B[0] - sum(B[j] for j in range(1, m))


def cf_filter(x, p_low=6.0, p_high=32.0, method="asymmetric", p=None,
              drift=True):
    r"""Extract the band :math:`[p_l, p_u]` by one of the three routes.

    Defaults are the quarterly business cycle, 6 to 32 quarters (1.5
    to 8 years).
    """
    if method not in _METHODS:
        raise ValueError("crfflt: method must be one of %s, got %r"
                         % (", ".join(_METHODS), method))
    v = [float(q) for q in k.vec(x)]
    T = len(v)
    if T < 5:
        raise ValueError("crfflt: need at least 5 observations, got %d"
                         % T)
    mu = 0.0
    if drift and method != "symmetric":
        da = drift_adjust(v)
        v, mu = da["adjusted"], da["drift"]
    B = ideal_weights(p_low, p_high, T)["B"]

    if method == "symmetric":
        pp = int(p) if p is not None else min(12, (T - 1) // 2)
        if pp < 1 or 2 * pp >= T:
            raise ValueError("crfflt: p must satisfy 1 <= p < T/2, "
                             "got %d for T = %d" % (pp, T))
        w = [B[j] for j in range(pp)]
        end = _tail(B, pp)
        out = [float("nan")] * T
        for t in range(pp, T - pp):
            s = w[0] * v[t]
            for j in range(1, pp):
                s += w[j] * (v[t + j] + v[t - j])
            s += end * (v[t + pp] + v[t - pp])
            out[t] = s
        wts = [end] + [w[j] for j in range(pp - 1, 0, -1)] + \
              [w[0]] + [w[j] for j in range(1, pp)] + [end]
        return RichResult(payload={
            "estimate": out, "cycle": out, "method": "symmetric",
            "p": pp, "weights": wts, "weight_sum": sum(wts),
            "n_missing": 2 * pp, "drift_removed": 0.0,
            "note": "two unit roots, so the output is invariant to "
                    "drift; costs the first and last p observations",
            "reference": "Christiano & Fitzgerald (2003) footnote 5",
        })

    if method == "one_sided":
        out = [float("nan")] * T
        for t in range(T):
            back = t
            if back < 2:
                continue
            s = 0.5 * B[0] * v[t]
            for j in range(1, back):
                s += B[j] * v[t - j]
            s += _tail(B, back) * v[0]
            out[t] = s
        return RichResult(payload={
            "estimate": out, "cycle": out, "method": "one_sided",
            "drift_removed": mu,
            "note": "eq. (1.4): current and past data only, for real "
                    "time estimation",
            "reference": "Christiano & Fitzgerald (2003) eq. (1.4)",
        })

    out, sums = [], []
    for t in range(T):
        f, b = T - 1 - t, t
        w = [0.0] * T
        # at an endpoint only one tail is available, so the centre
        # weight is halved -- eq. (1.4) -- and the single tail closes
        # the sum; in the interior both tails do, each absorbing half
        # of B_0 through the identity B_0 + 2 sum_j B_j = 0.
        w[t] += B[0] if (f >= 1 and b >= 1) else 0.5 * B[0]
        for j in range(1, f):
            w[t + j] += B[j]
        for j in range(1, b):
            w[t - j] += B[j]
        if f >= 1:
            w[T - 1] += _tail(B, f)
        if b >= 1:
            w[0] += _tail(B, b)
        out.append(sum(w[i] * v[i] for i in range(T)))
        sums.append(sum(w))
    return RichResult(payload={
        "estimate": out, "cycle": out, "method": "asymmetric",
        "trend": [v[i] - out[i] for i in range(T)],
        "drift_removed": mu, "weight_sums": sums,
        "max_abs_weight_sum": max(abs(s) for s in sums),
        "note": "eq. (1.2): time-varying weights, uses every "
                "observation; ONE unit root, so the data must be "
                "drift-adjusted first",
        "reference": "Christiano & Fitzgerald (2003) eqs. (1.2)-(1.3)",
    })


def frequency_response(weights, omega):
    r""":math:`\sum_j w_j e^{-i\omega j}` in magnitude, for a symmetric
    weight vector centred on its middle element.
    """
    w = [float(q) for q in k.vec(weights)]
    n = len(w)
    c = (n - 1) // 2
    re = sum(w[j] * math.cos(omega * (j - c)) for j in range(n))
    im = sum(-w[j] * math.sin(omega * (j - c)) for j in range(n))
    return math.sqrt(re * re + im * im)


def cheatsheet():
    return ("crfflt: band pass by PROJECTION, not truncation. Ideal "
            "weights B_j = (sin jb - sin ja)/(pi j), B_0 = (b-a)/pi, "
            "a = 2pi/p_u, b = 2pi/p_l. They decay SLOWLY -- still "
            "non-zero at j = 120 -- so truncation bites. The "
            "recommended filter is optimal under a RANDOM WALK "
            "assumption that is probably false and works anyway. "
            "Endpoint weights absorb the tails and every weight vector "
            "sums to ZERO, so a constant is annihilated exactly. "
            "Asymmetric has one unit root and needs drift adjustment; "
            "symmetric has two and does not.")


# compact alias per ledger/NAMING.md
christianofitzgerald = cf_filter
