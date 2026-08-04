"""Tests for rghfd.rangayyan_higuchi_fd.

Spec: Rangayyan, Biomedical Signal Analysis (Wiley/IEEE, 2024),
Sec. 5.13.2 "Higuchi's method", p. 304, eqs (5.39)-(5.41); after
Higuchi (1988), Physica D 31:277-283.

Equations transcribed from the typeset PDF, not the text extraction -- the
extraction renders (5.40) as a flat run of tokens with the fraction structure
lost, which is precisely where the off-by-one below was hiding.

The chapter has no numeric worked example for HFD, so the checks here are a
direct re-derivation of (5.40)/(5.41) from the equations plus the limiting
values the method is defined by.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.bsastat import rangayyan_higuchi_fd


def _hfd_reference(x, kmax):
    """Literal transcription of eqs (5.39)-(5.41), 1-based as printed.

    Deliberately written from the book rather than from the implementation,
    so agreement between the two is evidence rather than tautology.
    """
    x = np.asarray(x, float)
    N = x.size
    Lk = []
    for k in range(1, kmax + 1):
        Lmk = []
        for m in range(1, k + 1):           # (5.39): m = 1..k, 1-based
            n_i = (N - m) // k              # (5.40): floor((N-m)/k)
            if n_i < 1:
                continue
            s = sum(abs(x[m + i * k - 1] - x[m + (i - 1) * k - 1])
                    for i in range(1, n_i + 1))
            Lmk.append((1.0 / k) * ((N - 1) / (k * n_i)) * s)
        Lk.append(sum(Lmk) / len(Lmk))      # (5.41): (1/k) sum_m
    ks = np.arange(1, kmax + 1)
    return np.polyfit(np.log(1.0 / ks), np.log(Lk), 1)[0]


def test_matches_literal_transcription_of_equations_5_39_to_5_41():
    """Implementation == the printed equations, re-derived independently.

    This is what caught the defect: the loop ran m over 0..k-1 but fed that
    0-based index into the normaliser floor((N-m)/k), so the denominator was
    floor((N-m+1)/k) while the numerator still summed floor((N-m)/k) terms.
    The two disagreed whenever (N-m) was not a multiple of k.
    """
    rng = np.random.default_rng(20260726)
    for n, kmax in ((200, 8), (137, 6), (501, 10)):
        x = rng.standard_normal(n)
        got = rangayyan_higuchi_fd(x, kmax=kmax)["HFD"]
        assert np.isclose(got, _hfd_reference(x, kmax), rtol=1e-12, atol=1e-12)


def test_identity_straight_line_has_fd_near_one():
    """A straight line is topologically 1-D, so FD -> 1.

    The book's own reading of the statistic ("~1 smooth, ~2 rough"). A line
    is the one input whose answer is known exactly a priori.
    """
    x = np.linspace(0.0, 10.0, 400)
    assert abs(rangayyan_higuchi_fd(x, kmax=10)["HFD"] - 1.0) < 0.02


def test_identity_white_noise_has_fd_near_two():
    """Gaussian white noise fills the plane, so FD -> 2."""
    x = np.random.default_rng(7).standard_normal(2000)
    assert 1.85 < rangayyan_higuchi_fd(x, kmax=10)["HFD"] < 2.05


def test_identity_scale_invariance():
    """FD is dimensionless: it must not move under affine rescaling.

    L(m,k) is linear in the amplitude of x, so a constant factor shifts every
    log L(k) by the same amount and leaves the slope untouched. An offset
    cancels in the differences. A normalisation bug that made the scaling
    k-dependent would break this.
    """
    x = np.random.default_rng(11).standard_normal(600)
    base = rangayyan_higuchi_fd(x, kmax=10)["HFD"]
    for a, b in ((1000.0, 0.0), (0.001, 0.0), (1.0, 500.0), (-3.0, -2.0)):
        assert np.isclose(rangayyan_higuchi_fd(a * x + b, kmax=10)["HFD"], base,
                          rtol=1e-9, atol=1e-9)


def test_returns_documented_keys():
    """The generated test asserted an "estimate" key this never promised."""
    x = np.random.default_rng(1).standard_normal(300)
    result = rangayyan_higuchi_fd(x, kmax=8)
    for key in ("HFD", "log_L", "log_inv_k", "kmax"):
        assert key in result
    assert np.isfinite(result["HFD"])


def test_rejects_degenerate_input():
    """A single sample has no fractal dimension; the function is right to refuse.

    The old edge test passed np.array([42.0]) and then asserted result["n"] == 1
    -- expecting both a key the function does not return and a computation it
    cannot perform.
    """
    with pytest.raises(ValueError, match=r"len\(x\) >= 4"):
        rangayyan_higuchi_fd(np.array([42.0]))
    with pytest.raises(ValueError, match="kmax >= 2"):
        rangayyan_higuchi_fd(np.arange(50.0), kmax=1)
