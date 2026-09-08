# morie.fn -- function file (rootcoder007/morie)
"""Simulated power of the sign test from caller-supplied samples."""

import math

from ._richresult import RichResult

__all__ = ['signsimpow', 'gibbons_sign_simpower']


def signsimpow(samples, m0, kcrit):
    """Monte-Carlo power of the sign test over pre-drawn samples.

    Section 5.4.5 (book p. 175).  The book's MINITAB macro draws 1000
    samples under H1, computes K for each and reports the rejection
    fraction.  The draws are an *argument* here rather than an internal
    generator, so the estimate is reproducible across languages: pass
    the same matrix of samples and both arms return the same number.

    Parameters
    ----------
    samples : sequence of sequence of float
        One row per simulated sample.
    m0 : float
        Hypothesised median.
    kcrit : int
        Rejection region is K >= kcrit.

    Returns
    -------
    RichResult
        keys ``power``, ``rejections``, ``nsim``, ``kmean``,
        ``kcrit``, ``method``.

    References
    ----------
    Gibbons & Chakraborti (2011), Sec. 5.4.5, p. 175 (Table 5.4.2).
    """
    rows = [[float(v) for v in r] for r in samples]
    nsim = len(rows)
    if nsim < 1:
        raise ValueError("samples must be non-empty.")
    kcrit = int(kcrit)
    ks = [sum(1 for v in r if v > float(m0)) for r in rows]
    rej = sum(1 for k in ks if k >= kcrit)
    return RichResult(
        payload={
            "power": rej / nsim,
            "rejections": int(rej),
            "nsim": int(nsim),
            "kmean": sum(ks) / nsim,
            "kcrit": kcrit,
            "method": "simulated sign-test power over supplied samples",
        }
    )


gibbons_sign_simpower = signsimpow
