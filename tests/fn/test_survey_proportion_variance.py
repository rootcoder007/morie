"""Tests for survey_proportion_variance (Bilder & Loughin 2025, eq. 6.9)."""

import pytest

from morie.fn.survey_proportion_variance import survey_proportion_variance


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e9_basic():
    """pi_i = N_i/N linearises to (N_i - pi N)/N, whose variance is the
    quadratic form a' Sigma a with a = (1, -pi)/N and Sigma the
    covariance matrix of (N_i, N)."""
    v_ni, v_n, c = 400.0, 900.0, 450.0
    pi, N = 0.3, 1000.0
    a = [1.0 / N, -pi / N]
    S = [[v_ni, c], [c, v_n]]
    q = sum(a[i] * S[i][j] * a[j] for i in range(2) for j in range(2))
    r = survey_proportion_variance(v_ni, v_n, c, pi, N)
    assert r["value"] == pytest.approx(q, rel=1e-15)
    # (400 + 0.09*900 - 2*0.3*450)/1e6 = 211/1e6
    assert r["value"] == pytest.approx(2.11e-4, rel=1e-15)


def test_analysis_of_categorical_data_with_r_chapman_hall_crc_christo6e9_edge():
    """A known total (Var N = Cov = 0) reduces to Var(N_i)/N^2; a
    non-positive total raises."""
    assert survey_proportion_variance(25.0, 0.0, 0.0, 0.4, 50.0)["value"] == pytest.approx(0.01, rel=1e-15)
    with pytest.raises(ValueError):
        survey_proportion_variance(1.0, 1.0, 0.0, 0.5, 0.0)
