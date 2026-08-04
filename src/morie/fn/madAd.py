# morie.fn -- k02 batch (rootcoder007/morie)
"""Modified z-score anomaly detector.

Source consulted: Iglewicz, B. and Hoaglin, D.C. (1993), *How to Detect and
Handle Outliers*, ASQC Basic References in Quality Control vol. 16, chapter 5.
Their recommended statistic is

    M_i = 0.6745 (x_i - median(x)) / MAD,   MAD = median(|x - median(x)|)

with the rule "label x_i an outlier when |M_i| > 3.5".  The constant 0.6745 is
Phi^-1(3/4), so M_i is on the z-score scale.  Both the constant and the
threshold are the book's, and both are exposed as arguments.
"""

from __future__ import annotations

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["mad_anomaly_score"]


def mad_anomaly_score(x, threshold=3.5, constant=0.6745):
    """Iglewicz-Hoaglin modified z-scores.

    Parameters
    ----------
    x : array-like
        Sample.
    threshold : float, default 3.5
        Flagging cut-off on |M|.
    constant : float, default 0.6745
        Numerator constant, Phi^-1(3/4).

    Returns
    -------
    RichResult
        estimate (largest |M|), scores, outlier, center, mad, n_outliers,
        n, method.
    """
    v = np.asarray(x, dtype=float).ravel()
    ctr = float(np.median(v))
    mad = float(np.median(np.abs(v - ctr)))
    if mad > 0.0:
        m = float(constant) * (v - ctr) / mad
    else:
        m = np.zeros(len(v))
    flag = [bool(abs(float(t)) > float(threshold)) for t in m]
    return RichResult(
        payload={
            "estimate": float(np.max(np.abs(m))),
            "scores": m.tolist(),
            "outlier": flag,
            "center": ctr,
            "mad": mad,
            "n_outliers": int(sum(1 for t in flag if t)),
            "n": int(len(v)),
            "method": "Modified z-score outlier labelling (Iglewicz & Hoaglin 1993, ch. 5)",
        }
    )


# CANONICAL TEST
# >>> x = [2.1, 3.4, 1.9, 5.6, 2.8, 3.1, 9.9, 2.5, 3.3, 2.7]
# >>> r = mad_anomaly_score(x)
# >>> assert r["outlier"][6] is True and r["n_outliers"] == 1
# >>> assert abs(r["mad"] - 0.45) < 1e-14


def cheatsheet():
    return "madAd(x): Iglewicz-Hoaglin modified z-score anomaly scores."


madanomalyscore = mad_anomaly_score
