"""Tests for rng223 (Rangayyan Eq. 4.51 test signal)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.bsasig import rangayyan_ch4_test_signal_three_events


def test_rng223_matches_eq_4_51_sample_by_sample():
    """The nine nonzero samples of Eq. 4.51, written out."""
    x = np.asarray(rangayyan_ch4_test_signal_three_events(36)["signal"], dtype=float)
    want = {5: 3.0, 6: 2.0, 7: 1.0, 16: 1.5, 17: 1.0, 18: 0.5, 26: 0.75, 27: 0.5, 28: 0.25}
    for k, v in want.items():
        assert x[k] == pytest.approx(v, abs=1e-12), k
    assert np.count_nonzero(x) == 9
    assert x.size == 36


def test_rng223_is_scaled_shifts_of_the_basic_pattern():
    """Eq. 4.53: x = g(n-5) + 0.5 g(n-16) + 0.25 g(n-26)."""
    r = rangayyan_ch4_test_signal_three_events(40)
    x = np.asarray(r["signal"], dtype=float)
    g = np.asarray(r["pattern"], dtype=float)
    np.testing.assert_allclose(x[5:8], g, atol=1e-12)
    np.testing.assert_allclose(x[16:19], 0.5 * g, atol=1e-12)
    np.testing.assert_allclose(x[26:29], 0.25 * g, atol=1e-12)


def test_rng223_matched_filter_peaks_at_the_events():
    """The signal exists to illustrate matched filtering: correlating with
    g peaks exactly at the three event END positions with amplitudes in
    the 1 : 0.5 : 0.25 ratio."""
    r = rangayyan_ch4_test_signal_three_events(40)
    x = np.asarray(r["signal"], dtype=float)
    g = np.asarray(r["pattern"], dtype=float)
    corr = np.correlate(x, g, mode="full")
    # ACF(g) at lag 0 is 3^2 + 2^2 + 1^2 = 14, so the three matched-filter
    # peaks are exactly 14, 7 and 3.5 (amplitudes 1, 0.5, 0.25).
    assert float(corr.max()) == pytest.approx(14.0, abs=1e-12)
    assert np.isclose(corr, 7.0).any()
    assert np.isclose(corr, 3.5).any()


def test_rng223_rejects_a_length_that_cuts_the_last_event():
    with pytest.raises(ValueError, match="at least 29"):
        rangayyan_ch4_test_signal_three_events(20)
