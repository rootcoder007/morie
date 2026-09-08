# morie.fn -- function file (rootcoder007/morie)
"""Page-Hinkley change detector."""

from __future__ import annotations

from . import _t4core as T

from ._richresult import RichResult

__all__ = ["page_hinkley"]


def page_hinkley(x, threshold, delta=0.005, direction="increase"):
    """Page-Hinkley sequential test for a shift in the mean.

    Formula: with ``xbar_T`` the running mean of the first ``T``
    observations,

        ``m_T = sum_{t<=T} (x_t - xbar_t - delta)``,
        ``M_T = min_{t<=T} m_t``,
        ``PH_T = m_T - M_T``

    and a change is flagged at the first ``T`` with ``PH_T > lambda``.
    The mirrored form, ``max_{t<=T} m_t - m_T`` on ``x_t - xbar_t +
    delta``, detects a decrease.

    ``delta`` is the magnitude below which a deviation is treated as
    noise: anything smaller drives ``m_T`` downwards, so the detector
    stays quiet.  ``threshold`` trades detection delay against false
    alarms.  The running mean is updated online, not recomputed, so the
    statistic depends only on the data seen so far -- which is the whole
    point of a sequential scheme and also what makes the two language
    arms agree term by term.

    Parameters
    ----------
    x : array-like
        Stream in arrival order.
    threshold : float
        Alarm level ``lambda``.
    delta : float
        Tolerated magnitude of drift.
    direction : {"increase", "decrease"}
        Which shift to detect.

    Returns
    -------
    RichResult
        ``statistic`` (final PH), ``detected`` (bool), ``changepoint``
        (1-based index of the alarm, or 0), ``ph`` (the whole path),
        ``n``, ``method``.

    References
    ----------
    Page (1954), Continuous inspection schemes, Biometrika 41:100-115;
    Hinkley (1971), Inference about the change-point from cumulative sum
    tests, Biometrika 58:509-523.  Both paywalled at JSTOR (HTTP 403);
    the running-mean form used here is the one standardised in the
    concept-drift literature, Gama, Zliobaite, Bifet, Pechenizkiy and
    Bouchachia (2014), A survey on concept drift adaptation, ACM
    Computing Surveys 46(4):44, sec. 4.2 -- ``m_T = sum (x_t - xbar_t -
    delta)``, ``PH_T = m_T - min m_t``, alarm at ``PH_T > lambda``.
    """
    x = T.vec(x)
    n = len(x)
    if n == 0:
        raise ValueError("empty stream")
    if direction not in ("increase", "decrease"):
        raise ValueError("direction must be 'increase' or 'decrease'")
    sign = 1.0 if direction == "increase" else -1.0
    delta = float(delta)
    run = 0.0
    m = 0.0
    ext = 0.0
    path = []
    detected = False
    cp = 0
    for t in range(n):
        run += x[t]
        xbar = run / (t + 1.0)
        m += sign * (x[t] - xbar) - delta
        if t == 0 or m < ext:
            ext = m
        ph = m - ext
        path.append(ph)
        if not detected and ph > threshold:
            detected = True
            cp = t + 1
    return RichResult(
        payload={
            "statistic": float(path[-1]),
            "detected": bool(detected),
            "changepoint": int(cp),
            "ph": path,
            "n": int(n),
            "method": "Page-Hinkley change detector",
        }
    )


def cheatsheet():
    return "page_hinkley(x, threshold, delta): PH_T = m_T - min m_t."


# compact alias per ledger/NAMING.md
pagehinkley = page_hinkley
