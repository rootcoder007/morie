"""2-parameter logistic IRT model (Birnbaum 1968)."""

from __future__ import annotations

from ._irtcore import INF, broadcast, expit, seq_
from ._richresult import RichResult

__all__ = ["two_parameter_logistic"]


def two_parameter_logistic(theta, a=1.0, b=0.0):
    r"""Probability of a correct response under the two-parameter logistic model.

    .. math::
        P_i(\theta) = \frac{1}{1 + \exp(-a_i(\theta - b_i))}

    The previous body was a placeholder: it averaged a leading ``y``
    argument and never touched ``theta``, ``a`` or ``b``, so it returned
    the same number for every item. That spurious ``y`` appears nowhere in
    the formula and is gone.

    Parameters
    ----------
    theta : float or array-like
        Ability values.
    a : float or array-like, default 1.0
        Discrimination. Must be finite. ``a = 1`` gives the Rasch model.
    b : float or array-like, default 0.0
        Difficulty, on the theta scale.

    Returns
    -------
    RichResult
        ``p``, ``logit`` (the linear predictor :math:`a(\theta - b)`),
        ``theta``, ``a``, ``b``, ``n``, ``method``.

    Notes
    -----
    Samejima writes this model as
    :math:`P(\theta) = 1/(1 + e^{-Da(\theta-b)})` with the scaling constant
    :math:`D`; the logistic metric used here is :math:`D = 1`. Fold ``D``
    into ``a`` if you want the normal metric (``D = 1.702``).

    References
    ----------
    Birnbaum, A. (1968). Some latent trait models and their use in inferring
    an examinee's ability. In F. M. Lord & M. R. Novick, *Statistical
    Theories of Mental Test Scores*, chs. 17-20.

    Samejima, F. (1969). *Estimation of Latent Ability Using a Response
    Pattern of Graded Scores*. Psychometric Monograph No. 17. Eq. (10-13),
    p. 79 states the logistic model for dichotomous items and attributes
    the sufficient statistic to "Birnbaum in Lord & Novick, 1968".
    """
    th = [float(v) for v in seq_(theta)]
    n = len(th)
    if n == 0:
        raise ValueError("theta is empty.")
    av = broadcast(a, n, "a")
    bv = broadcast(b, n, "b")
    for v in av:
        if v != v or v in (INF, -INF):
            raise ValueError("a must be finite; got %r" % (v,))

    logit = [av[i] * (th[i] - bv[i]) for i in range(n)]
    p = [expit(z) for z in logit]

    return RichResult(
        payload={
            "p": p,
            "logit": logit,
            "theta": th,
            "a": av,
            "b": bv,
            "n": n,
            "method": "Two-parameter logistic IRT model (Birnbaum 1968)",
        }
    )


def cheatsheet():
    return "irt2pl: 2PL  P(theta) = 1/(1+exp(-a(theta-b)))"
