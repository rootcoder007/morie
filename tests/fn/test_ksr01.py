"""Tests for ksr01.kosorok_empirical_process."""

from morie.fn import _array_core as np
import pytest

from morie.fn.ksr01 import kosorok_empirical_process


def _emp_mean(x, t):
    """Compute the empirical cdf value F_n(t) = mean(x <= t) by hand."""
    n = len(x)
    return sum(1 for v in x if v <= t) / n


def test_ksr01_identity_function_gives_root_n_mean_deviation():
    """At a single evaluation point t = mu0 with F(t) = identity(t)/range,
    G_n(t) = sqrt(n)(F_n(t) - F(t)) reduces to a computable form
    -- specifically, sqrt(n)*(mean(x <= t) - F(t))."""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    t = np.array([2.5])
    F = np.array([0.5])
    r = kosorok_empirical_process(x, t, F)
    Fn = _emp_mean(x, float(t[0]))
    expected_Gn = np.sqrt(4) * (Fn - float(F[0]))
    assert float(r["Gn"][0]) == pytest.approx(expected_Gn, rel=1e-12)
    assert float(r["n"]) == 4
    assert float(r["k"]) == 1


def test_ksr01_centered_data_gives_zero():
    """For a sample that is symmetric about 0 and t = 0 with F(0) = 0.5,
    G_n(0) = sqrt(n)(F_n(0) - 0.5) -- with x = {-2,-1,1,2}, F_n(0) = 0.5."""
    x = np.array([-2.0, -1.0, 1.0, 2.0])
    t = np.array([0.0])
    F = np.array([0.5])
    r = kosorok_empirical_process(x, t, F)
    Fn = _emp_mean(x, 0.0)
    expected_Gn = np.sqrt(4) * (Fn - 0.5)
    assert float(r["Gn"][0]) == pytest.approx(expected_Gn, abs=1e-12)
    assert float(r["Gn"][0]) == pytest.approx(0.0, abs=1e-12)


def test_ksr01_scales_as_root_n():
    """Duplicating the sample doubles n. With t and F fixed at the same
    evaluation point, F_n is unchanged so G_n scales by sqrt(2)."""
    x = np.array([1.0, 3.0, 5.0])
    t = np.array([2.0])
    F = np.array([1.0 / 3.0])
    a = float(kosorok_empirical_process(x, t, F)["Gn"][0])
    b = float(kosorok_empirical_process(np.tile(x, 2), t, F)["Gn"][0])
    assert b == pytest.approx(np.sqrt(2) * a, rel=1e-12)
