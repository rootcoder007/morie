"""3-parameter logistic IRT model with a guessing asymptote."""

from __future__ import annotations

from ._irtcore import INF, broadcast, expit, seq_
from ._richresult import RichResult

__all__ = ["three_parameter_logistic"]


def three_parameter_logistic(theta, a=1.0, b=0.0, c=0.0):
    r"""Probability of a correct response under the three-parameter logistic model.

    .. math::
        P_i(\theta) = c_i + \frac{1 - c_i}{1 + \exp(-a_i(\theta - b_i))}

    The previous body was a placeholder: it averaged a leading ``y``
    argument and never referenced ``theta``, ``a``, ``b`` or ``c``. That
    spurious ``y`` is gone.

    At :math:`\theta = b` the curve passes exactly through
    :math:`(1 + c)/2`, and the asymptotes are :math:`c` (as
    :math:`\theta \to -\infty`) and 1. ``c = 0`` recovers the 2PL and
    ``a = 1, c = 0`` the Rasch model.

    Parameters
    ----------
    theta : float or array-like
        Ability values.
    a : float or array-like, default 1.0
        Discrimination; must be finite.
    b : float or array-like, default 0.0
        Difficulty.
    c : float or array-like, default 0.0
        Lower asymptote (pseudo-guessing), in [0, 1).

    Returns
    -------
    RichResult
        ``p``, ``logit``, ``theta``, ``a``, ``b``, ``c``, ``n``, ``method``.

    References
    ----------
    Birnbaum, A. (1968). Some latent trait models and their use in inferring
    an examinee's ability. In F. M. Lord & M. R. Novick, *Statistical
    Theories of Mental Test Scores*, chs. 17-20.

    Lord, F. M. (1980). *Applications of Item Response Theory to Practical
    Testing Problems*. Erlbaum.

    Samejima, F. (1969). *Estimation of Latent Ability Using a Response
    Pattern of Graded Scores*. Psychometric Monograph No. 17, eq. (10-13),
    p. 79 (the two-parameter logistic kernel of this curve).
    """
    th = [float(v) for v in seq_(theta)]
    n = len(th)
    if n == 0:
        raise ValueError("theta is empty.")
    av = broadcast(a, n, "a")
    bv = broadcast(b, n, "b")
    cv = broadcast(c, n, "c")

    for i in range(n):
        if not (cv[i] >= 0.0) or cv[i] >= 1.0:
            raise ValueError("c must lie in [0, 1); got %r" % (cv[i],))
        if av[i] != av[i] or av[i] in (INF, -INF):
            raise ValueError("a must be finite; got %r" % (av[i],))

    logit = [av[i] * (th[i] - bv[i]) for i in range(n)]
    p = [cv[i] + (1.0 - cv[i]) * expit(logit[i]) for i in range(n)]

    return RichResult(
        payload={
            "p": p,
            "logit": logit,
            "theta": th,
            "a": av,
            "b": bv,
            "c": cv,
            "n": n,
            "method": "Three-parameter logistic IRT model (Birnbaum 1968)",
        }
    )


def cheatsheet():
    return "irt3pl: 3PL  P(theta) = c + (1-c)/(1+exp(-a(theta-b)))"
