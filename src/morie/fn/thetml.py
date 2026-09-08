"""Maximum likelihood estimate of the IRT ability parameter theta."""

from __future__ import annotations

import math

from ._irtcore import INF, broadcast, expit, seq_
from ._richresult import RichResult

__all__ = ["theta_mle"]

_LOWER = -6.0
_UPPER = 6.0
_NGRID = 1201
_MAXIT = 60


def theta_mle(x, a=1.0, b=0.0, c=0.0):
    r"""Maximum likelihood estimate of ability from one response pattern.

    For dichotomous responses :math:`x_j \in \{0, 1\}` to :math:`n` items
    with 3PL curves :math:`P_j(\theta)`, the log-likelihood is

    .. math::
        \ell(\theta) = \sum_j x_j \log P_j(\theta)
                     + (1 - x_j) \log\{1 - P_j(\theta)\}

    with score and Fisher information

    .. math::
        \ell'(\theta) = \sum_j \frac{(x_j - P_j) P'_j}{P_j Q_j},
        \qquad
        I(\theta) = \sum_j \frac{(P'_j)^2}{P_j Q_j},

    the second being Samejima's eq. (6-9) summed over items. Here
    :math:`P'_j = a_j (1 - c_j) P^{*}_j Q^{*}_j`.

    The previous body was a placeholder: it averaged a leading ``X``
    argument and never referenced the item parameters at all, and the
    second argument ``items`` was never used. Both are gone.

    The maximiser is found deterministically -- a fixed 1201-point grid on
    [-6, 6] to pick the global mode (the 3PL likelihood can be multimodal),
    then 60 Fisher-scoring steps with the step clamped to +/- 0.5. No
    randomness and no adaptive stopping, so the Python and R arms visit the
    identical sequence of iterates.

    Parameters
    ----------
    x : array-like
        Response pattern of 0/1 scores, one per item.
    a : float or array-like, default 1.0
        Discriminations, length 1 or ``len(x)``.
    b : float or array-like, default 0.0
        Difficulties.
    c : float or array-like, default 0.0
        Lower asymptotes, in [0, 1).

    Returns
    -------
    RichResult
        ``theta``, ``se`` (:math:`1/\sqrt{I(\hat\theta)}`), ``loglik``,
        ``score``, ``information``, ``raw_score``, ``n_items``,
        ``converged``, ``method``.

    Notes
    -----
    A perfect or a zero response pattern has no finite maximiser: the
    likelihood increases monotonically towards the boundary. Those cases
    return ``+inf`` / ``-inf`` with ``converged = False``, rather than the
    grid endpoint dressed up as an estimate.

    Anchor: with the Rasch parameterisation (all ``a = 1``, all ``b = 0``,
    ``c = 0``) the score equation is :math:`r - n\,\mathrm{expit}(\theta) = 0`,
    so :math:`\hat\theta = \log\{r/(n-r)\}` in closed form.

    References
    ----------
    Birnbaum, A. (1968). In F. M. Lord & M. R. Novick, *Statistical Theories
    of Mental Test Scores*, chs. 17-20.

    Samejima, F. (1969). *Estimation of Latent Ability Using a Response
    Pattern of Graded Scores*. Psychometric Monograph No. 17, eq. (6-9)
    p. 39 and eq. (10-15) p. 79.
    """
    xs = [float(v) for v in seq_(x)]
    n = len(xs)
    if n == 0:
        raise ValueError("x is empty.")
    for v in xs:
        if v not in (0.0, 1.0):
            raise ValueError("x must contain only 0 and 1; got %r" % (v,))
    av = broadcast(a, n, "a")
    bv = broadcast(b, n, "b")
    cv = broadcast(c, n, "c")
    for i in range(n):
        if not (cv[i] >= 0.0) or cv[i] >= 1.0:
            raise ValueError("c must lie in [0, 1); got %r" % (cv[i],))
        if av[i] != av[i] or av[i] in (INF, -INF):
            raise ValueError("a must be finite; got %r" % (av[i],))

    r = sum(xs)

    def parts(t):
        ll = 0.0
        sc = 0.0
        fi = 0.0
        for i in range(n):
            ps = expit(av[i] * (t - bv[i]))
            p = cv[i] + (1.0 - cv[i]) * ps
            q = 1.0 - p
            d = av[i] * (1.0 - cv[i]) * ps * (1.0 - ps)
            ll += xs[i] * math.log(p) + (1.0 - xs[i]) * math.log(q)
            sc += (xs[i] - p) * d / (p * q)
            fi += d * d / (p * q)
        return ll, sc, fi

    if r == 0.0 or r == n:
        t = INF if r == n else -INF
        return RichResult(
            payload={
                "theta": t,
                "se": INF,
                "loglik": 0.0 if all(cv[i] == 0.0 for i in range(n)) else parts(
                    _UPPER if r == n else _LOWER)[0],
                "score": float("nan"),
                "information": 0.0,
                "raw_score": r,
                "n_items": n,
                "converged": False,
                "method": "MLE of theta, 3PL (Birnbaum 1968)",
            }
        )

    step = (_UPPER - _LOWER) / (_NGRID - 1)
    best = _LOWER
    bestll = parts(_LOWER)[0]
    for k in range(1, _NGRID):
        t = _LOWER + k * step
        ll = parts(t)[0]
        if ll > bestll:
            bestll = ll
            best = t

    t = best
    for _ in range(_MAXIT):
        _, sc, fi = parts(t)
        if fi <= 0.0:
            break
        d = sc / fi
        if d > 0.5:
            d = 0.5
        elif d < -0.5:
            d = -0.5
        t = t + d

    ll, sc, fi = parts(t)
    return RichResult(
        payload={
            "theta": t,
            "se": 1.0 / math.sqrt(fi) if fi > 0.0 else INF,
            "loglik": ll,
            "score": sc,
            "information": fi,
            "raw_score": r,
            "n_items": n,
            "converged": abs(sc) < 1e-8,
            "method": "MLE of theta, 3PL (Birnbaum 1968)",
        }
    )


def cheatsheet():
    return "thetml: MLE of theta from a 0/1 response pattern under the 3PL"


# compact alias per ledger/NAMING.md
thetamle = theta_mle
