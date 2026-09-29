"""Tests for morie.fn.ensrv: recompute from the definition."""

from morie.fn.ensrv import ensemble_variance

X = [2.5, -1.0, 4.25, 0.5, 3.0, -2.75, 1.5, 6.0]


def test_bessel_variance():
    S = [X, [v * 0.5 for v in X], [v - 1 for v in X]]
    r = ensemble_variance(S)
    for j in range(8):
        c = [row[j] for row in S]
        m = sum(c) / 3
        assert abs(r.value[j] - sum((t - m) ** 2 for t in c) / 2) < 1e-14
    assert abs(r.extra["mean_var"] - sum(r.value) / 8) < 1e-15
