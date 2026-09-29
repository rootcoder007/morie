"""Tests for olsfit (Hedderich eq 3.96 / 6.25)."""

import pytest

from morie.fn.olsfit import olsfit


def test_book_example_lung_cancer_on_asbestos():
    """Hedderich p. 132: lm(lungca ~ asbestos) prints 0.54047 and 0.01772."""
    x = [50, 400, 500, 900, 1100, 1600, 1800, 2000, 3000]
    y = [2, 6, 5, 10, 26, 42, 37, 28, 50]
    n = 9
    sxy_raw = n * sum(a * b for a, b in zip(x, y)) - sum(x) * sum(y)
    sxx_raw = n * sum(a * a for a in x) - sum(x) ** 2
    b = sxy_raw / sxx_raw
    a = sum(y) / n - b * sum(x) / n
    r = olsfit(x, y)
    assert r["slope"] == pytest.approx(b, rel=1e-12)
    assert r["intercept"] == pytest.approx(a, rel=1e-10)
    assert round(r["intercept"], 5) == 0.54047 and round(r["slope"], 5) == 0.01772
    assert sum(r["residuals"]) == pytest.approx(0.0, abs=1e-10)
