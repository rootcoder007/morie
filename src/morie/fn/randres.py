# morie.fn -- function file (rootcoder007/morie)
"""Warner randomized response estimator."""

import math

from . import _s04core as S
from . import _tail1core as C

from ._richresult import RichResult

__all__ = ["randomized_response"]


def randomized_response(y, truth=None, p=0.7):
    """Recover a population rate from deliberately noised answers.

    The respondent answers one of two questions chosen by a private
    randomiser, so no individual answer reveals anything -- which is
    exactly what makes people answer honestly about stigmatised
    behaviour.  The population rate is still identified because the
    noise mechanism is known.  The estimator blows up as ``p`` nears one
    half, since at exactly one half the answer carries no information at
    all.

    Formula: with ``P(yes) = p pi + (1 - p)(1 - pi)``,
    ``pi_hat = (lambda_hat - (1 - p)) / (2p - 1)`` and
    ``Var(pi_hat) = lambda(1 - lambda) / (n (2p - 1)^2)``.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed yes/no answers.
    truth : array-like, optional
        True statuses, when known; only used to report the gap.
    p : float, default 0.7
        Probability the randomiser selected the sensitive question.

    Returns
    -------
    RichResult
        ``estimate`` (``pi_hat``), ``se``, ``lambda`` (raw yes rate),
        ``truth_rate``, ``n``.

    References
    ----------
    Warner, S. L. (1965).  Randomized response: a survey technique for
    eliminating evasive answer bias.  Journal of the American
    Statistical Association 60:63-69, equations (1) and (3).
    """
    v = C.vec(y)
    n = len(v)
    lam = sum(v) / n
    p = float(p)
    d = 2.0 * p - 1.0
    pi = (lam - (1.0 - p)) / d if d != 0.0 else float("nan")
    var = lam * (1.0 - lam) / (n * d * d) if d != 0.0 else float("nan")
    tr = sum(C.vec(truth)) / n if truth is not None else float("nan")
    return RichResult(payload={
        "estimate": pi, "se": math.sqrt(var) if var == var and var >= 0 else float("nan"),
        "lambda": lam, "truth_rate": tr, "n": n,
        "method": "Warner randomized response estimator"})


def cheatsheet():
    return "randres: Warner randomized response estimator."
