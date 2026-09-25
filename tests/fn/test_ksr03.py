"""ksr03: Glivenko-Cantelli / KS sup|F_n - F| statistic.

Kosorok, M. R. (2008). *Introduction to Empirical Processes and
Semiparametric Inference*, Ch. 2 -- in the library, filed under its ISBN
(978-0-387-74978-5) rather than its title.
"""

import math

import pytest
import pytest

from morie.fn.ksr03 import kosorok_glivenko_cantelli as gc


X = [math.sin(1.7 * i) * 2 for i in range(25)]


def _Phi(x, s):
    return 0.5 * (1 + math.erf(x / (s * math.sqrt(2))))


def test_ksr03_basic():
    """D_n = max(D+, D-) equals scipy.stats.kstest against N(0, 1.5^2)."""
    r = gc(X, [_Phi(x, 1.5) for x in X])
    assert r["statistic"] == pytest.approx(0.12875393936155266, rel=1e-12)
    assert r["dkw_bound"] == pytest.approx(min(1.0, 2 * math.exp(-2 * 25 * r["statistic"] ** 2)), rel=1e-15)
    assert sorted(X)[int(r["argmax"]) - 1] == pytest.approx(1.5466557791324413, rel=1e-12)


def test_ksr03_edge():
    """A sample whose F values sit at the midpoints (i - 1/2)/n has D = 1/(2n)."""
    n = 8
    r = gc(list(range(n)), [(i + 0.5) / n for i in range(n)])
    assert r["statistic"] == pytest.approx(1 / (2 * n), rel=1e-14)


