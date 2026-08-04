# morie.fn -- function file (rootcoder007/morie)
"""Mantel-Haenszel summary incidence rate ratio over strata.

Rothman & Greenland, Modern Epidemiology, with the variance of
Greenland, S. & Robins, J.M. (1985), Estimation of a common effect
parameter from sparse follow-up data, Biometrics 41:55-68.  The
textbook is not in the corpus and the publisher blocks direct fetch,
so the estimator and its standard error were taken verbatim from the
OpenEpi "Comparing Two Person-Time Rates" technical documentation,
read in full:

    IRR_MH = sum_i (a_i T0_i / T_i) / sum_i (b_i T1_i / T_i)

    SE(ln IRR_MH) = sqrt( sum_i m_i T1_i T0_i / T_i^2 )
                    / sqrt( [sum_i a_i T0_i / T_i]
                            [sum_i b_i T1_i / T_i] )

with a_i and b_i the exposed and unexposed case counts in stratum i,
T1_i and T0_i the corresponding person-times, T_i = T1_i + T0_i and
m_i = a_i + b_i the stratum case total.
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["mantel_haenszel_rate"]

_Z = {0.90: 1.6448536269514722, 0.95: 1.959963984540054,
      0.99: 2.5758293035489004}


def _zcrit(confidence):
    c = float(confidence)
    for k, v in _Z.items():
        if abs(c - k) < 1e-12:
            return v
    raise ValueError("confidence must be one of 0.90, 0.95, 0.99")


def _stratum(s):
    """Accept either a mapping with keys a, T1, b, T0 or a positional
    sequence (a, T1, b, T0)."""
    if hasattr(s, "get"):
        return (float(s["a"]), float(s["T1"]),
                float(s["b"]), float(s["T0"]))
    t = list(s)
    if len(t) != 4:
        raise ValueError("each stratum needs (a, T1, b, T0)")
    return (float(t[0]), float(t[1]), float(t[2]), float(t[3]))


def mantel_haenszel_rate(strata, confidence=0.95):
    """Pooled rate ratio across strata, weighting each stratum by its
    person-time so that sparse strata contribute without being
    dropped.

    Parameters
    ----------
    strata : sequence of (a, T1, b, T0) or mappings with those keys --
        exposed cases, exposed person-time, unexposed cases,
        unexposed person-time.
    confidence : 0.90, 0.95 or 0.99.

    Returns
    -------
    RichResult with keys estimate, ln_estimate, se_ln, ci_lower,
    ci_upper, numerator, denominator, n_strata, confidence, method.

    References
    ----------
    Rothman & Greenland, Modern Epidemiology; Greenland & Robins
    (1985) Biometrics 41:55-68.
    """
    rows = [_stratum(s) for s in strata]
    if not rows:
        raise ValueError("need at least one stratum")
    num = den = vnum = 0.0
    for a, T1, b, T0 in rows:
        T = T1 + T0
        if T <= 0:
            raise ValueError("stratum person-time must be positive")
        num += a * T0 / T
        den += b * T1 / T
        vnum += (a + b) * T1 * T0 / (T * T)
    if den <= 0 or num <= 0:
        raise ValueError("both arms need cases for a rate ratio")
    irr = num / den
    se = math.sqrt(vnum) / math.sqrt(num * den)
    z = _zcrit(confidence)
    return with_describe_pointer(RichResult(payload={
        "estimate": float(irr), "ln_estimate": float(math.log(irr)),
        "se_ln": float(se),
        "ci_lower": float(irr * math.exp(-z * se)),
        "ci_upper": float(irr * math.exp(z * se)),
        "numerator": float(num), "denominator": float(den),
        "n_strata": len(rows), "confidence": float(confidence),
        "method": "Mantel-Haenszel rate ratio (Greenland & Robins 1985)",
    }), "mhrate")


def cheatsheet():
    return "mhrate: Mantel-Haenszel summary incidence rate ratio"


# compact alias per ledger/NAMING.md
mhraterr = mantel_haenszel_rate
