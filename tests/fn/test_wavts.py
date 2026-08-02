"""Tests for wavts.wavelet_time_series.

Spec: Percival, D. B. & Walden, A. T. (2000). Wavelet Methods for Time Series
Analysis. Cambridge University Press. DWT p.56; jth level wavelet detail D_j
p.64; multiresolution analysis p.65; energy (squared norm) E_X pp.42, 72.

The book gives no small numeric worked example that can be transcribed
verbatim, so the anchors here are the structural identities an orthonormal DWT
must satisfy -- energy preservation above all -- plus a hand-computed Haar
transform, which is short enough to write out by hand from the definition.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.wavts import wavelet_time_series


def test_haar_one_level_matches_a_hand_computation():
    """One Haar level, computed by hand from the definition.

    For pairs (x0,x1),(x2,x3),... the orthonormal Haar filters give
    approximation (x_even + x_odd)/sqrt(2) and detail (x_even - x_odd)/sqrt(2).
    Short enough to write out, so this is a genuine independent check rather
    than a restatement of the implementation.
    """
    x = np.array([1.0, 3.0, 2.0, 6.0])
    res = wavelet_time_series(x, wavelet="haar", level=1)
    s = np.sqrt(2.0)
    want_a = np.array([(1 + 3) / s, (2 + 6) / s])
    want_d = np.array([(1 - 3) / s, (2 - 6) / s])
    assert np.allclose(np.sort(np.abs(res["approximation"])), np.sort(np.abs(want_a)))
    assert np.allclose(np.sort(np.abs(res["details"][0])), np.sort(np.abs(want_d)))


def test_identity_energy_is_preserved():
    """P&W pp.42, 72: the orthonormal DWT preserves E_X = sum x_t^2.

    This is the identity that breaks first if the filter normalisation drifts
    -- for instance if a 1/2 crept in where 1/sqrt(2) belongs, which would
    still return plausible-looking coefficients.
    """
    for n in (64, 128, 256):
        x = np.random.default_rng(n).standard_normal(n)
        res = wavelet_time_series(x, wavelet="haar")
        assert np.isclose(sum(res["energies"]), float(np.sum(x**2)), rtol=1e-12)


def test_energies_are_aligned_with_the_returned_arrays():
    """energies[0] is the approximation and energies[i+1] is details[i].

    They were misaligned: details came back deepest-first while energies ran
    shallowest-first, so every level past the first reported the energy of a
    different band than the coefficients beside it. Nothing raised, and the
    totals still summed correctly, which is why it survived.
    """
    x = np.random.default_rng(11).standard_normal(64)
    res = wavelet_time_series(x, wavelet="haar", level=3)
    assert np.isclose(res["energies"][0], float(np.sum(res["approximation"] ** 2)))
    for i, d in enumerate(res["details"]):
        assert np.isclose(res["energies"][i + 1], float(np.sum(d**2))), \
            f"energies[{i + 1}] does not match details[{i}]"


def test_details_are_ordered_deepest_first():
    """P&W index D_j by level; pywt returns [cA_n, cD_n, ..., cD_1].

    The native path must match that ordering, or swapping PyWavelets in or out
    would silently reverse the caller's levels.
    """
    x = np.random.default_rng(13).standard_normal(64)
    res = wavelet_time_series(x, wavelet="haar", level=3)
    sizes = [d.size for d in res["details"]]
    assert sizes == sorted(sizes), f"expected coarsest-first, got sizes {sizes}"


def test_refuses_a_non_haar_wavelet_without_pywavelets():
    """Asking for db4 without PyWavelets must raise, not return Haar.

    The previous code wrapped the whole pywt path in `except Exception: pass`,
    so a request for db4 -- or a typo -- fell through to the Haar fallback and
    returned coefficients in the wrong basis with no warning.
    """
    pytest.importorskip  # noqa: B018  (documented below)
    import importlib.util
    if importlib.util.find_spec("pywt") is not None:
        pytest.skip("PyWavelets installed; the no-pywt guard cannot be exercised")
    x = np.random.default_rng(17).standard_normal(64)
    for bad in ("db4", "sym8", "coif3", "not_a_wavelet"):
        with pytest.raises(ValueError, match="needs PyWavelets"):
            wavelet_time_series(x, wavelet=bad)


def test_haar_still_works_without_pywavelets():
    """The native path covers Haar, so the base install stays useful."""
    x = np.random.default_rng(19).standard_normal(64)
    res = wavelet_time_series(x, wavelet="haar")
    assert np.isclose(sum(res["energies"]), float(np.sum(x**2)), rtol=1e-12)


def test_energy_preserved_for_every_family_when_pywavelets_present():
    """Orthonormality must not depend on which family is chosen.

    This is what caught the boundary-mode bug: under pywt's default
    "symmetric" extension the transform is redundant and energy is NOT
    preserved, so db4/sym8/coif3 all failed while native Haar passed.
    """
    pytest.importorskip("pywt")
    x = np.random.default_rng(31).standard_normal(256)
    for w in ("haar", "db4", "sym8", "coif3"):
        res = wavelet_time_series(x, wavelet=w, level=3)
        assert np.isclose(sum(res["energies"]), float(np.sum(x**2)), rtol=1e-9), \
            f"energy not preserved for {w}"


def test_pywt_and_native_haar_agree():
    """Installing PyWavelets must not change the answer for Haar.

    Both paths use the periodized, orthonormal DWT that P&W define, so the
    optional dependency is an extension of coverage, not a change of result.
    """
    pytest.importorskip("pywt")
    x = np.random.default_rng(37).standard_normal(64)
    viapywt = wavelet_time_series(x, wavelet="haar", level=3)
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **k):
        if name == "pywt":
            raise ImportError("blocked for this test")
        return real(name, *a, **k)

    builtins.__import__ = blocked
    try:
        native = wavelet_time_series(x, wavelet="haar", level=3)
    finally:
        builtins.__import__ = real
    assert np.allclose(sorted(viapywt["energies"]), sorted(native["energies"]))


def test_identity_linearity():
    """The DWT is linear: T(ax) = a T(x)."""
    x = np.random.default_rng(23).standard_normal(64)
    base = wavelet_time_series(x, wavelet="haar", level=2)
    for a in (7.0, -3.0, 0.25):
        got = wavelet_time_series(a * x, wavelet="haar", level=2)
        assert np.allclose(got["approximation"], a * base["approximation"])
        for g, b in zip(got["details"], base["details"]):
            assert np.allclose(g, a * b)


def test_rejects_series_shorter_than_four():
    """The old edge test passed a single sample and asserted result["n"] == 1
    -- a computation the function cannot perform."""
    with pytest.raises(ValueError, match="at least 4 observations"):
        wavelet_time_series(np.array([42.0]))


def test_returns_documented_keys():
    """The old basic test asserted an "estimate" key this never promised."""
    res = wavelet_time_series(np.random.default_rng(29).standard_normal(64))
    for key in ("approximation", "details", "energies", "level", "n", "wavelet", "method"):
        assert key in res
