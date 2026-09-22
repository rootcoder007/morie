"""Tests for gb1231.gibbons_page_test."""

import math

import pytest

from morie.fn import _array_core as np

from morie.fn.gb1231 import gibbons_page_test


def test_gb1231_basic():
    """Test basic functionality with a 5x4 block design."""
    rng = np.random.default_rng(42)
    # k=5 blocks, n=4 treatments: data shape (k, n).
    data = rng.normal(0, 1, (5, 4))

    result = gibbons_page_test(data)
    assert isinstance(result, dict)
    for key in ("statistic", "z", "p_value", "rav",
                "rank_sums", "k", "n", "method"):
        assert key in result

    assert result["k"] == 5
    assert result["n"] == 4
    assert result["method"].startswith("Page")

    # Independently recompute L from the literature formula.
    rows = [[float(v) for v in r] for r in data]
    k = len(rows)
    n = len(rows[0])
    Y = [float(i + 1) for i in range(n)]
    rsum = [0.0] * n
    for r in rows:
        order = sorted(range(n), key=lambda i: r[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and r[order[j + 1]] == r[order[i]]:
                j += 1
            mid = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                rk[order[t]] = mid
            i = j + 1
        for j in range(n):
            rsum[j] += rk[j]
    ell_expected = sum(Y[j] * rsum[j] for j in range(n))
    assert result["statistic"] == pytest.approx(ell_expected)
    assert result["rank_sums"] == pytest.approx(rsum)
    assert result["rank_sums"] == pytest.approx(rsum)


def test_gb1231_weights():
    """Test that user-supplied weights override the default 1..n ranking."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (4, 3))

    weights = [3.0, 2.0, 1.0]
    result = gibbons_page_test(data, weights=weights)

    rows = [[float(v) for v in r] for r in data]
    k = len(rows)
    n = len(rows[0])
    rsum = [0.0] * n
    for r in rows:
        order = sorted(range(n), key=lambda i: r[i])
        rk = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and r[order[j + 1]] == r[order[i]]:
                j += 1
            mid = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                rk[order[t]] = mid
            i = j + 1
        for j in range(n):
            rsum[j] += rk[j]
    ell_expected = sum(weights[j] * rsum[j] for j in range(n))
    assert result["statistic"] == pytest.approx(ell_expected)

    # z and rav from the docstring formulas.
    z_expected = (12.0 * (ell_expected - 0.5)
                  - 3.0 * k * n * (n + 1.0) ** 2) / (
        n * (n + 1.0) * math.sqrt(k * (n - 1.0)))
    rav_expected = (12.0 * ell_expected / (k * (float(n) ** 3 - n))
                    - 3.0 * (n + 1.0) / (n - 1.0))
    assert result["z"] == pytest.approx(z_expected)
    assert result["rav"] == pytest.approx(rav_expected)


def test_gb1231_edge():
    """Test edge cases: monotone increasing data yields large L."""
    # Strictly monotone treatments -> perfect ordering -> large L and z.
    data = [[1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0]]
    result = gibbons_page_test(data)

    assert isinstance(result, dict)
    assert result["k"] == 3
    assert result["n"] == 3
    # Each block contributes ranks 1,2,3; rsum = [3, 6, 9]; Y = 1,2,3.
    assert result["rank_sums"] == pytest.approx([3.0, 6.0, 9.0])
    assert result["statistic"] == pytest.approx(3 * 1 + 6 * 2 + 9 * 3)

    # Verify edge-case arithmetic on the literature formula.
    ell = result["statistic"]
    k, n = 3, 3
    z_expected = (12.0 * (ell - 0.5)
                  - 3.0 * k * n * (n + 1.0) ** 2) / (
        n * (n + 1.0) * math.sqrt(k * (n - 1.0)))
    rav_expected = (12.0 * ell / (k * (float(n) ** 3 - n))
                    - 3.0 * (n + 1.0) / (n - 1.0))
    assert result["z"] == pytest.approx(z_expected)
    assert result["rav"] == pytest.approx(rav_expected)
