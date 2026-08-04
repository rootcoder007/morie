"""Item information function (Birnbaum)."""

from __future__ import annotations

from ._irtcore import INF, broadcast, expit, seq_
from ._richresult import RichResult

__all__ = ["item_information"]


def item_information(theta, a=1.0, b=0.0, c=0.0):
    r"""Fisher information contributed by one dichotomous item.

    Samejima (1969) eq. (6-9), p. 39, gives the information function of a
    dichotomously scored item as

    .. math::
        I_g(\theta) = \frac{\{P'_g(\theta)\}^2}{P_g(\theta) Q_g(\theta)}

    and names it (p. 40) "the item information function ... by Birnbaum".
    For the three-parameter logistic curve
    :math:`P = c + (1-c)P^{*}`, :math:`P^{*} = 1/(1+e^{-a(\theta-b)})`, the
    derivative is :math:`P' = a(1-c)P^{*}Q^{*}`, so

    .. math::
        I(\theta) = \frac{a^2 (1-c)^2 (P^{*}Q^{*})^2}{P\,Q}.

    With ``c = 0`` this collapses to the familiar
    :math:`I(\theta) = a^2 P(\theta) Q(\theta)`, which is the form the old
    placeholder docstring quoted while the body averaged a spurious leading
    ``y`` argument and ignored every item parameter. That ``y`` is gone.

    Parameters
    ----------
    theta : float or array-like
        Ability values.
    a : float or array-like, default 1.0
        Discrimination; must be finite.
    b : float or array-like, default 0.0
        Difficulty.
    c : float or array-like, default 0.0
        Lower asymptote, in [0, 1).

    Returns
    -------
    RichResult
        ``info`` (per theta), ``p``, ``dp`` (:math:`P'`), ``theta``, ``a``,
        ``b``, ``c``, ``n``, ``total`` (the test information
        :math:`\sum_g I_g` when the inputs describe several items at one
        ability) and ``method``.

    References
    ----------
    Samejima, F. (1969). *Estimation of Latent Ability Using a Response
    Pattern of Graded Scores*. Psychometric Monograph No. 17. Eq. (6-9),
    p. 39; naming on p. 40; eq. (10-15), p. 79 for :math:`P' = D a P Q`.

    Birnbaum, A. (1968). In F. M. Lord & M. R. Novick, *Statistical Theories
    of Mental Test Scores*, chs. 17-20.
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

    p = []
    dp = []
    info = []
    for i in range(n):
        ps = expit(av[i] * (th[i] - bv[i]))
        qs = 1.0 - ps
        pi = cv[i] + (1.0 - cv[i]) * ps
        qi = 1.0 - pi
        d = av[i] * (1.0 - cv[i]) * ps * qs
        p.append(pi)
        dp.append(d)
        info.append(d * d / (pi * qi))

    return RichResult(
        payload={
            "info": info,
            "p": p,
            "dp": dp,
            "theta": th,
            "a": av,
            "b": bv,
            "c": cv,
            "n": n,
            "total": sum(info),
            "method": "Item information function (Birnbaum; Samejima 1969 eq. 6-9)",
        }
    )


def cheatsheet():
    return "iinfo: item information  I(theta) = P'(theta)^2 / (P Q); a^2 P Q when c = 0"
