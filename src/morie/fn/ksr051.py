# morie.fn -- function file (rootcoder007/morie)
"""Continuous invertibility."""

from . import _array_core as np

from ._richresult import RichResult

__all__ = ["kosorok_ch2_continuous_invertibility"]


def kosorok_ch2_continuous_invertibility(A, theta_1, theta_2=None, c=None,
                                         n_pairs=200, rng=None, radius=1.0):
    r"""Continuous invertibility condition:

    .. math:: \|A(\theta_1) - A(\theta_2)\|_L
              \ge c\,\|\theta_1 - \theta_2\|
              \quad \text{for some } c > 0.

    A lower Lipschitz bound. It is what guarantees that consistency of
    :math:`A(\hat\theta)` transfers back to consistency of
    :math:`\hat\theta` -- without it an estimator can drive the
    criterion to zero while wandering in the parameter, which is the
    classic identifiability failure.

    Estimates the largest valid c by minimising the ratio over sampled
    pairs, so the returned value is an UPPER bound on the true
    constant (a finite sample cannot find the worst pair).

    Parameters
    ----------
    A : callable
        The map.
    theta_1 : array-like
        Base point, or one of an explicit pair with ``theta_2``.
    theta_2 : array-like, optional
        Second point for a single-pair evaluation.
    c : float, optional
        A candidate constant to test against.
    n_pairs : int, default 200
        Random pairs when theta_2 is omitted.
    rng : numpy Generator, optional
    radius : float, default 1.0
        Sampling radius around theta_1.

    Returns
    -------
    RichResult
        keys: ``c_estimate``, ``is_upper_bound`` (True),
        ``min_ratio``, ``holds_for_c`` (if c given), ``n_pairs``,
        ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Ch. 2 (continuous invertibility of maps).
    """
    t1 = np.atleast_1d(np.asarray(theta_1, dtype=float))
    if theta_2 is not None:
        t2 = np.atleast_1d(np.asarray(theta_2, dtype=float))
        d = float(np.linalg.norm(t1 - t2))
        if d == 0:
            raise ValueError("theta_1 and theta_2 must differ.")
        ratio = float(np.linalg.norm(
            np.asarray(A(t1), dtype=float) - np.asarray(A(t2), dtype=float))) / d
        return RichResult(
            payload={"c_estimate": ratio, "is_upper_bound": True,
                     "min_ratio": ratio,
                     "holds_for_c": None if c is None else bool(ratio >= float(c)),
                     "n_pairs": 1,
                     "method": "||A(t1) - A(t2)||/||t1 - t2|| at the supplied pair"}
        )
    rng = np.random.default_rng(0) if rng is None else rng
    ratios = []
    for _ in range(int(n_pairs)):
        u = rng.standard_normal(t1.size)
        u = u / max(np.linalg.norm(u), 1e-12) * rng.uniform(1e-3, float(radius))
        t2 = t1 + u
        d = float(np.linalg.norm(u))
        ratios.append(float(np.linalg.norm(
            np.asarray(A(t1), dtype=float) - np.asarray(A(t2), dtype=float))) / d)
    ratios = np.array(ratios)
    lo = float(ratios.min())
    return RichResult(
        payload={"c_estimate": lo, "is_upper_bound": True, "min_ratio": lo,
                 "holds_for_c": None if c is None else bool(lo >= float(c)),
                 "n_pairs": int(n_pairs),
                 "method": "min over sampled pairs => UPPER bound on the true c"}
    )


def cheatsheet():
    return "ksr051: lower Lipschitz bound; sampled min is an UPPER bound on c"
