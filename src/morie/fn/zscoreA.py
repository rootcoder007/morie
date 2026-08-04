# morie.fn -- function file (rootcoder007/morie)
"""Z-score anomaly flagging.

Basic descriptive screening.  Triage confirmed this names no owning
source; the standard rule is implemented and no citation is
manufactured.
"""

import math

from ._richresult import RichResult, with_describe_pointer

__all__ = ["zscore_anomaly"]


def zscore_anomaly(x, k, ddof=1):
    """Flag points whose standardized distance from the mean exceeds k,

        |x_i - mu| / sigma > k.

    Sensitivity warning, not a defect: both mu and sigma are computed
    from the same series being screened, so a large outlier inflates
    sigma and masks itself.  The rule is a screen, not a test, and a
    robust centre and scale should be preferred when the contamination
    is more than a point or two.  A constant series has sigma zero and
    yields no flags rather than dividing by zero.

    Parameters
    ----------
    x : array-like series to screen.
    k : float, the threshold in standard deviations.
    ddof : int, delta degrees of freedom of the standard deviation.

    Returns
    -------
    RichResult with keys estimate (the count of flagged points),
    z, flags, indices, mean, sd, k, n, method.
    """
    v = [float(t) for t in x]
    n = len(v)
    if n < 2:
        raise ValueError("need at least two observations")
    kk = float(k)
    mu = sum(v) / n
    den = n - int(ddof)
    sd = math.sqrt(sum((t - mu) ** 2 for t in v) / den) if den > 0 else 0.0
    if sd > 0:
        z = [abs(t - mu) / sd for t in v]
    else:
        z = [0.0] * n
    flags = [t > kk for t in z]
    return with_describe_pointer(RichResult(payload={
        "estimate": float(sum(1 for t in flags if t)), "z": z,
        "flags": flags,
        "indices": [i for i in range(n) if flags[i]],
        "mean": float(mu), "sd": float(sd), "k": kk, "n": n,
        "method": "z-score anomaly flagging",
    }), "zscoreA")


def cheatsheet():
    return "zscoreA: Z-score anomaly"


# compact alias per ledger/NAMING.md
zanomaly = zscore_anomaly
