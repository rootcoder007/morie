"""Anchored tests for kcusum (Harchaoui-Moulines-Bach 2008 KCpA)."""

import math

from morie.fn.kcusum import kcusum, kernel_cusum

X1 = [0.1, -0.2, 0.05, 0.3, -0.1, 5.2, 4.9, 5.1, 5.3, 4.8,
      1.9, 2.1, 2.0, 1.8, 2.2]


def test_linear_kernel_closed_form():
    # For the linear kernel in 1-d the KFDR at k reduces to the
    # closed form (k(n-k)/n) (mu2-mu1)^2 / (sigma_W^2 + gamma) with
    # n sigma_W^2 = k sigma_1^2 + (n-k) sigma_2^2 (biased variances),
    # d1 = sigma_W^2/(sigma_W^2+gamma), d2 = d1^2 (their Sec. 3
    # Remark: linear kernel recovers the classical mean-change
    # statistic). Hand-computed at k = 5, gamma = 0.1:
    # kfdr = 24.1817679339887, T = 17.4683476431183.
    r = kcusum(X1, kernel="linear", gamma=0.1, kmin=2, kmax=13)
    assert r["estimate"] == 5
    assert abs(r["kfdr"] - 24.1817679339887) < 1e-9
    assert abs(r["statistic"] - 17.4683476431183) < 1e-9
    n, k, g = 15, 5, 0.1
    mu1 = sum(X1[:k]) / k
    mu2 = sum(X1[k:]) / (n - k)
    v1 = sum((v - mu1) ** 2 for v in X1[:k]) / k
    v2 = sum((v - mu2) ** 2 for v in X1[k:]) / (n - k)
    sw = (k * v1 + (n - k) * v2) / n
    kfdr = (k * (n - k) / n) * (mu2 - mu1) ** 2 / (sw + g)
    d1 = sw / (sw + g)
    d2 = d1 * d1
    assert abs(r["kfdr"] - kfdr) < 1e-10
    assert abs(r["d1"] - d1) < 1e-10
    assert abs(r["d2"] - d2) < 1e-10
    assert abs(r["statistic"] - (kfdr - d1) / math.sqrt(2 * d2)) < 1e-9


def test_gaussian_kernel_locates_big_change():
    r = kcusum(X1, kernel="gaussian", gamma=0.1)
    assert r["estimate"] == 5
    assert r["statistic"] > 5.0


def test_threshold_decision():
    r = kcusum(X1, kernel="linear", gamma=0.1, threshold=1e6)
    assert r["detected"] is False
    r = kcusum(X1, kernel="linear", gamma=0.1, threshold=1.0)
    assert r["detected"] is True


def test_alias():
    a = kernel_cusum(X1, kernel="linear", gamma=0.1)
    b = kcusum(X1, kernel="linear", gamma=0.1)
    assert a["statistic"] == b["statistic"]
