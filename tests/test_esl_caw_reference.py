"""Curds and whey (ESL 3.72-3.75) against the cancor-based computation in R."""

import math

from morie.fn.eslcaw import esl_curds_whey


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_curds_whey():
    i = range(1, 51)
    X = [[math.sin(t), math.cos(2 * t), math.log(t) / 3, ((7 * t) % 11) / 11] for t in i]
    Y = [
        [
            r[0] + 0.5 * r[1] + 0.3 * math.cos(9 * t),
            r[1] - r[2] + 0.4 * math.sin(7 * t),
            0.2 * r[3] + 0.5 * math.cos(5 * t),
        ]
        for r, t in zip(X, i)
    ]
    r = esl_curds_whey(X, Y)
    assert all(
        close(a, b)
        for a, b in zip(r["canonical_correlations"], (0.97072296565740290, 0.91365197315831137, 0.14572730766932465))
    )
    assert all(
        close(a, b) for a, b in zip(r["shrinkage"], (0.99512550202875527, 0.98441093037738459, 0.21335116212293923))
    )
    ref = (
        (0.59262826575703398, -0.26519461158987256, 0.066123358864783405),
        (0.57026424073910253, -0.75573089739836019, 0.060069557697641676),
    )
    assert all(close(a, b, 1e-11) for fr, rr in zip(r["fitted"][:2], ref) for a, b in zip(fr, rr))
    h = esl_curds_whey(X, Y, lambda_=2.0)  # eq 3.75, ridge hybrid
    assert all(
        close(a, b, 1e-11)
        for a, b in zip(h["fitted"][0], (0.568551217461193303, -0.574153022003884184, 0.071178066550125443))
    )
