"""Item characteristic curve, three-parameter logistic (Birnbaum 1968)."""

from __future__ import annotations

import math

from ._richresult import RichResult

__all__ = ["item_characteristic_curve"]


def item_characteristic_curve(theta, a=1.0, b=0.0, c=0.0):
    r"""Probability of a correct response under the 3PL model.

    .. math::
        P_i(\theta) = c_i + \frac{1 - c_i}{1 + \exp(-a_i(\theta - b_i))}

    with discrimination :math:`a_i`, difficulty :math:`b_i` and lower
    asymptote (pseudo-guessing) :math:`c_i`. Setting c = 0 gives the 2PL,
    and a = 1, c = 0 gives the Rasch/1PL model.

    The previous body was a placeholder: it took the mean and standard
    error of a leading ``y`` argument and never referenced ``theta``,
    ``a``, ``b`` or ``c`` at all, so it returned the same number whatever
    item parameters it was given. That spurious ``y`` is gone -- it
    appears nowhere in the formula the docstring states.

    ``a``, ``b`` and ``c`` may be scalars (applied to every theta) or
    sequences of the same length as ``theta``, one item per ability.

    Parameters
    ----------
    theta : array-like
        Ability values.
    a : float or array-like, default 1.0
        Discrimination. Must be finite; may be negative for a reverse-keyed
        item.
    b : float or array-like, default 0.0
        Difficulty, on the same scale as theta.
    c : float or array-like, default 0.0
        Lower asymptote, in [0, 1).

    Returns
    -------
    RichResult
        ``p`` (probabilities), ``theta``, ``a``, ``b``, ``c``, ``n``,
        ``logit`` (the linear predictor a(theta - b)) and ``method``.

    References
    ----------
    Birnbaum, A. (1968). Some latent trait models and their use in
    inferring an examinee's ability. In F. M. Lord & M. R. Novick,
    *Statistical Theories of Mental Test Scores*, chs. 17-20. Eq. (17.4.5)
    gives the three-parameter form.
    """
    th = [float(v) for v in _seq(theta)]
    n = len(th)
    if n == 0:
        raise ValueError("theta is empty.")
    av = _broadcast(a, n, "a")
    bv = _broadcast(b, n, "b")
    cv = _broadcast(c, n, "c")

    for i in range(n):
        if not (cv[i] >= 0.0) or cv[i] >= 1.0:
            raise ValueError("c must lie in [0, 1); got %r" % (cv[i],))
        if av[i] != av[i] or av[i] in (_INF, -_INF):
            raise ValueError("a must be finite; got %r" % (av[i],))

    logit = [av[i] * (th[i] - bv[i]) for i in range(n)]
    p = [cv[i] + (1.0 - cv[i]) * _expit(logit[i]) for i in range(n)]

    return RichResult(
        payload={
            "p": p,
            "logit": logit,
            "theta": th,
            "a": av,
            "b": bv,
            "c": cv,
            "n": n,
            "method": "Item characteristic curve, 3PL (Birnbaum 1968)",
        }
    )


_INF = float("inf")


def _seq(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (int, float)):
        return [x]
    return list(x)


def _broadcast(v, n, name):
    vals = [float(u) for u in _seq(v)]
    if len(vals) == 1:
        return vals * n
    if len(vals) != n:
        raise ValueError("%s has length %d; expected 1 or %d to match theta"
                         % (name, len(vals), n))
    return vals


def _expit(z):
    """1/(1+exp(-z)), written so neither tail overflows."""
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def cheatsheet():
    return ("icrf: item characteristic curve, 3PL "
            "P(theta) = c + (1-c)/(1+exp(-a(theta-b)))")


icrf = item_characteristic_curve
