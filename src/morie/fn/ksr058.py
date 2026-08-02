# morie.fn -- function file (rootcoder007/morie)
"""Law of the iterated logarithm for the empirical process."""

from . import _array_core as np

from ._kosorok import sup_norm
from ._richresult import RichResult

__all__ = ["kosorok_ch2_law_iterated_logarithm"]


def kosorok_ch2_law_iterated_logarithm(X=None, n=None, F=None, rng=None):
    r"""LIL for the empirical process, Kosorok eq. (2.21)
    (PDF-verified):

    .. math:: \limsup_{n\to\infty}
              \frac{\|G_n\|_\infty}{\sqrt{2\log\log n}}
              \le \frac12 \quad \text{a.s.},

    with equality when 1/2 lies in the range of F. The book also
    states Chung's companion,
    :math:`\liminf_n \sqrt{2\log\log n}\,\|G_n\|_\infty
    = \pi/2` a.s., which is returned alongside because the two
    together bracket the oscillation rather than only bounding it
    above.

    Returns the finite-n ratio, which for any realistic n sits well
    below 1/2 -- log log n grows so slowly that the LIL constant is
    essentially unobservable at sample sizes anyone runs. That gap is
    reported rather than hidden.

    Parameters
    ----------
    X : array-like, optional
        Sample; simulated uniform if omitted.
    n : int, optional
        Simulation size when X is omitted.
    F : callable, optional
        True CDF; uniform if omitted.
    rng : numpy Generator, optional
        For the simulated case.

    Returns
    -------
    RichResult
        keys: ``sup_norm``, ``lil_ratio``, ``lil_bound`` (0.5),
        ``chung_liminf_constant`` (pi/2), ``loglog_term``, ``n``,
        ``method``.
    References
    ----------
    Kosorok, M. R. (2008). *Introduction to Empirical Processes and
    Semiparametric Inference*. Springer. Eq. (2.21) and the Chung liminf that follows it.
    """
    if X is None:
        n = 1000 if n is None else int(n)
        rng = np.random.default_rng(0) if rng is None else rng
        X = rng.random(n)
    X = np.asarray(X, dtype=float).ravel()
    n = X.size
    if n < 16:
        raise ValueError(f"need at least 16 observations, got {n}.")
    ll = np.log(np.log(n))
    if ll <= 0:
        raise ValueError("n too small for log log n to be positive.")
    s = sup_norm(X, F)
    denom = np.sqrt(2.0 * ll)
    return RichResult(
        payload={"sup_norm": s, "lil_ratio": float(s / denom), "lil_bound": 0.5,
                 "chung_liminf_constant": float(np.pi / 2),
                 "loglog_term": float(denom), "n": int(n),
                 "method": "||G_n||_inf / sqrt(2 log log n) vs the 1/2 bound (eq. 2.21)"}
    )


def cheatsheet():
    return "ksr058: eq. (2.21) bound 1/2; Chung liminf pi/2; both reported"
