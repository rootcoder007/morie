"""Tests for rgadp.rangayyan_adaptive_filter.

Spec: Rangayyan & Krishnan (2024) Sec 3.10.2 "The least-mean-squares
adaptive filter", p.184; Widrow & Stearns (1985). The point of an adaptive
noise canceller is that the primary input minus the filtered reference has
less noise than the primary input did -- that is what is pinned here.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.rgadp import rangayyan_adaptive_filter


def test_rgadp_cancels_noise_correlated_with_the_reference():
    rng = np.random.default_rng(41)
    n = 4000
    t = np.arange(n) / 200.0
    clean = np.sin(2 * np.pi * 3.0 * t)
    noise = rng.standard_normal(n)
    # reference is a filtered copy of the same noise -- correlated with the
    # interference in the primary input, uncorrelated with `clean`
    reference = np.convolve(noise, np.array([0.6, 0.3, 0.1]), mode="same")
    primary = clean + noise
    r = rangayyan_adaptive_filter(primary, reference, mu=0.01, order=16)
    out = np.asarray(r["signal"], dtype=float)
    tail = slice(n // 2, None)
    err_before = float(np.mean((primary[tail] - clean[tail]) ** 2))
    err_after = float(np.mean((out[tail] - clean[tail]) ** 2))
    assert err_after < err_before


def test_rgadp_leaves_an_uncorrelated_reference_roughly_alone():
    rng = np.random.default_rng(42)
    n = 2000
    primary = np.sin(2 * np.pi * 0.01 * np.arange(n))
    reference = rng.standard_normal(n)
    out = np.asarray(
        rangayyan_adaptive_filter(primary, reference, mu=0.001, order=8)["signal"], dtype=float
    )
    # with nothing to cancel, the output must still track the primary
    assert float(np.corrcoef(out[n // 2 :], primary[n // 2 :])[0, 1]) > 0.9


def test_rgadp_reports_the_weights_it_converged_to():
    rng = np.random.default_rng(43)
    r = rangayyan_adaptive_filter(rng.standard_normal(1000), rng.standard_normal(1000), order=12)
    w = np.asarray(r["weights"], dtype=float)
    assert w.size == 12
    assert np.all(np.isfinite(w))
    assert r["order"] == 12


def test_rgadp_output_decomposes_the_primary_input():
    # signal + noise_estimate must reconstruct the primary input exactly:
    # the canceller subtracts its estimate, nothing else.
    rng = np.random.default_rng(44)
    x = rng.standard_normal(500)
    ref = rng.standard_normal(500)
    r = rangayyan_adaptive_filter(x, ref, order=8)
    recon = np.asarray(r["signal"], dtype=float) + np.asarray(r["noise_estimate"], dtype=float)
    assert np.allclose(recon, x, atol=1e-9)


def test_rgadp_records_the_step_size():
    rng = np.random.default_rng(45)
    r = rangayyan_adaptive_filter(rng.standard_normal(200), rng.standard_normal(200), mu=0.005)
    assert r["mu"] == pytest.approx(0.005)
