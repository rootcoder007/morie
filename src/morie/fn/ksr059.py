# morie.fn -- function file (rootcoder007/morie)
"""KMT strong approximation."""

import numpy as np

from ._kosorok import sup_norm
from ._richresult import RichResult

__all__ = ["kosorok_ch2_kmt_strong_approximation"]


def kosorok_ch2_kmt_strong_approximation(n, x=1.0, a=None, b=None, c=None,
                                         G_n=None, B_n=None, F=None):
    r"""KMT (Komlos-Major-Tusnady) strong approximation bound:

    .. math:: P\Big(\|G_n - B_n(F)\|_\infty >
              \frac{a\log n + x}{\sqrt n}\Big) \le b\,e^{-cx}.

    Far stronger than weak convergence: it puts the empirical process
    and a Brownian bridge on the SAME probability space within
    :math:`O(\log n/\sqrt n)`, rather than merely matching their
    distributions. That rate is what licenses simultaneous confidence
    bands.

    The constants a, b, c are universal but their numerical values are
    not stated in the text; they must be supplied, and omitting them
    raises rather than substituting invented numbers.

    Parameters
    ----------
    n : int
        Sample size.
    x : float, default 1.0
        Deviation parameter.
    a, b, c : float
        The universal constants. All three are required.
    G_n, B_n, F : array-like, optional
        Realisations; when both G_n and B_n are given their observed
        sup distance is compared against the threshold.

    Returns
    -------
    RichResult
        keys: ``threshold``, ``probability_bound``,
        ``observed_sup_distance`` (if given), ``within_threshold``,
        ``n``, ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (the KMT construction).
    """
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}.")
    if a is None or b is None or c is None:
        raise ValueError(
            "the KMT constants a, b and c are universal but their values are "
            "not given in the text; supply them explicitly."
        )
    a, b, c, x = float(a), float(b), float(c), float(x)
    thr = (a * np.log(n) + x) / np.sqrt(n)
    obs, within = None, None
    if G_n is not None and B_n is not None:
        g = np.asarray(G_n, dtype=float).ravel()
        bb = np.asarray(B_n, dtype=float).ravel()
        if g.shape != bb.shape:
            raise ValueError("G_n and B_n must have the same shape.")
        obs = float(np.max(np.abs(g - bb)))
        within = bool(obs <= thr)
    return RichResult(
        payload={"threshold": float(thr),
                 "probability_bound": float(min(1.0, b * np.exp(-c * x))),
                 "observed_sup_distance": obs, "within_threshold": within,
                 "n": n,
                 "method": "KMT: ||G_n - B_n(F)||_inf <= (a log n + x)/sqrt(n) w.h.p."}
    )


def cheatsheet():
    return "ksr059: same probability space, O(log n/sqrt n); constants must be supplied"
