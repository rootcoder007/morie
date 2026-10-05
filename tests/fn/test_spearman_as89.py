"""spearman_rho's p-value is R's cor.test(method = "spearman") default (AS 89), as rmorie.

Round-8 parity finding: Python gave the t approximation (0.9637059) where rmorie and base R give
the AS 89 value (0.9644038). Reference values: cor.test(x, y, method = "spearman")$p.value, R 4.6.
"""

import pytest

from morie.fn.rho import _prho, spearman_rho

X = [5.1, 6.3, 4.8, 7.2, 5.9, 6.6, 5.4, 6.0, 5.5, 6.8]
Y = [4.2, 5.0, 3.9, 5.8, 4.4, 4.9, 5.3, 4.1, 4.6, 5.2]
Z = [6.1, 5.2, 6.9, 7.4, 6.0, 5.8, 7.1, 6.6, 6.2, 6.4]
A = [1, 3, 2, 5, 4, 7, 6]
B = [2, 1, 4, 3, 7, 5, 6]
S = list(range(1, 31))
T = [(v * 7) % 31 for v in S]


@pytest.mark.parametrize(
    ("x", "y", "want"),
    [
        (X, Z, 0.86475352880452594),  # n = 10, Edgeworth, rho < 0 (upper tail of S)
        (X, Y, 0.054445067937541794),  # n = 10, Edgeworth, rho > 0 (lower tail of S)
        (A, B, 0.23571428571428571),  # n = 7, exact enumeration
        (A, B[::-1], 0.0027777777777777779),  # n = 7, exact, rho < 0
        (S, T, 0.63635017058467291),  # n = 30, lower tail
        (S, [-v for v in T], 0.63635017058467291),  # n = 30, upper tail
    ],
)
def test_matches_cor_test(x, y, want):
    assert spearman_rho(x, y)["p_value"] == pytest.approx(want, rel=1e-13)


def test_lower_tail_subtracts_the_edgeworth_term():
    # P[S < s] + P[S >= s] = 1 in the Edgeworth branch (the series term enters with opposite signs)
    for n, s in ((10, 120.0), (30, 3500.0), (200, 1.2e6)):
        assert _prho(n, s, True) + _prho(n, s, False) == pytest.approx(1.0, abs=1e-15)


def test_ties_fall_back_to_t_approximation():
    # cor.test with ties: exact is turned off, the t_{n-2} approximation is used
    r = spearman_rho([1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 1, 3, 3, 5, 4, 7, 6, 9, 8, 10])
    assert r["p_value"] == pytest.approx(6.3195162603542134e-06, rel=1e-12)
