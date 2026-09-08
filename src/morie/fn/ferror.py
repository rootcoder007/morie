# morie.fn -- function file (rootcoder007/morie)
"""Renewal-equation forecast."""

from ._richresult import RichResult

__all__ = ["fader_renewable"]


def fader_renewable(incidence, Rt, gen_int):
    """
    Renewal-equation forecast

    Formula: I_t = R_t sum_{s>=1} I_{t-s} w_s, where w is the generation
    interval distribution (normalised to sum to one) and
    Lambda_t = sum_s I_{t-s} w_s is the total infectiousness at time t.
    Forward projection feeds each simulated incidence back into
    Lambda for the following step.  The same identity read backwards
    gives the instantaneous reproduction number on the observed record,
    Rhat_t = I_t / Lambda_t.

    Parameters
    ----------
    incidence : array-like
        Observed incidence I_1..I_n.
    Rt : float or array-like
        Reproduction number for each forecast step; a scalar is held
        constant over a one-step horizon.
    gen_int : array-like
        Generation-interval weights w_1..w_S at lags 1..S; non-negative,
        renormalised internally.

    Returns
    -------
    result : dict
        Keys: estimate (total forecast incidence), forecast, lambda_,
        total, Rt_implied, horizon, n, method.

    References
    ----------
    Fraser (2007), PLoS ONE 2(8):e758, doi:10.1371/journal.pone.0000758.
    """
    inc = [float(v) for v in incidence]
    n = len(inc)
    if n == 0:
        raise ValueError("empty input: incidence has no observations")
    if any(v < 0.0 for v in inc):
        raise ValueError("incidence must be non-negative")
    w = [float(v) for v in ([gen_int] if isinstance(gen_int, (int, float)) else gen_int)]
    S = len(w)
    if S == 0:
        raise ValueError("gen_int must have at least one lag")
    if any(v < 0.0 for v in w):
        raise ValueError("gen_int weights must be non-negative")
    sw = sum(w)
    if sw <= 0.0:
        raise ValueError("gen_int weights must not all be zero")
    w = [v / sw for v in w]
    R = [float(Rt)] if isinstance(Rt, (int, float)) else [float(v) for v in Rt]
    H = len(R)
    if H == 0:
        raise ValueError("Rt must have at least one step")
    if any(v < 0.0 for v in R):
        raise ValueError("Rt must be non-negative")
    hist = list(inc)
    fc = []
    lam = []
    for k in range(H):
        L = 0.0
        for s in range(1, S + 1):
            j = len(hist) - s
            if j >= 0:
                L += hist[j] * w[s - 1]
        lam.append(L)
        v = R[k] * L
        fc.append(v)
        hist.append(v)
    Rimp = []
    for t in range(S, n):
        L = sum(inc[t - s] * w[s - 1] for s in range(1, S + 1))
        Rimp.append(inc[t] / L if L > 0.0 else float("nan"))
    return RichResult(payload={
        "estimate": sum(fc),
        "forecast": fc,
        "lambda_": lam,
        "total": sum(fc),
        "Rt_implied": Rimp,
        "horizon": H,
        "n": n,
        "method": "Renewal-equation forecast",
    })


def cheatsheet():
    return "ferror: Renewal-equation forecast"


# compact alias per ledger/NAMING.md
faderrenewable = fader_renewable
