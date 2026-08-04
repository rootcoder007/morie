"""Test cross_correlation (xcorr)."""

from morie.fn import _array_core as np
import pytest

from morie.fn._containers import DescriptiveResult
from morie.fn.ccf import cross_correlation


class TestCrossCorrelation:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(128)
        y = np.random.default_rng(43).standard_normal(128)
        result = cross_correlation(x, y)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "cross_correlation"

    def test_autocorrelation_peak(self):
        x = np.random.default_rng(42).standard_normal(128)
        result = cross_correlation(x, x, max_lag=10)
        assert result.value == pytest.approx(1.0, abs=0.01)

    def test_correlation_array(self):
        x = np.random.default_rng(42).standard_normal(64)
        y = np.random.default_rng(43).standard_normal(64)
        result = cross_correlation(x, y, max_lag=5)
        assert "correlation" in result.extra
        assert len(result.extra["correlation"]) == 11

    def test_the_bare_name_xcorr_now_means_the_RAW_cross_correlation(self):
        """This module no longer exports `xcorr`.  In signal processing
        xcorr is the RAW cross-correlation, so the bare name belongs to
        Rangayyan's R_xy(m) in bsacorr -- this one is normalized and keeps
        the explicit name."""
        import morie.fn as fn
        from morie.fn.bsacorr import xcorr as raw

        assert fn.xcorr is raw
        assert not hasattr(
            __import__("morie.fn.ccf", fromlist=["x"]), "xcorr")
        # and the two are genuinely different quantities: scaling y scales
        # the raw correlation and leaves the normalized one alone.
        import math
        x = [math.sin(i / 6.0) for i in range(50)]
        y = [math.sin((i - 3) / 6.0) for i in range(50)]
        y10 = [10.0 * v for v in y]

        r1 = max(raw(x, y, maxlag=5)["ccf"])
        r10 = max(raw(x, y10, maxlag=5)["ccf"])
        assert abs(r10 - 10.0 * r1) < 1e-9          # raw: scales with y

        n1 = cross_correlation(x, y, max_lag=5).extra["correlation"]
        n10 = cross_correlation(x, y10, max_lag=5).extra["correlation"]
        assert max(abs(a - b) for a, b in zip(n1, n10)) < 1e-9   # invariant
        assert max(abs(v) for v in n1) <= 1.0 + 1e-12            # bounded


def test_cross_correlation_is_the_ccf_with_the_lag_axis_reversed():
    """This module used to hold a SECOND implementation of fn/ccf.py -- the
    same numbers with the lag axis reversed, i.e. the opposite sign
    convention for the delay.  It now delegates; this pins the numbers so
    the delegation cannot silently change them."""
    import math

    from morie.fn.ccf import ccf

    x = [math.sin(i / 7.0) for i in range(60)]
    y = [math.sin((i - 4) / 7.0) + 0.1 * math.cos(i) for i in range(60)]
    got = list(cross_correlation(x, y, max_lag=10).extra["correlation"])
    want = list(ccf(x, y, nlags=10)["ccf_values"])[::-1]
    assert len(got) == 21
    for a, b in zip(got, want):
        assert abs(a - b) < 1e-12


def test_ccf_and_ccfn_run_at_all():
    """Both raised TypeError on every call -- np.arange yields floats here
    and a float cannot slice.  ccf died on any negative lag, ccf_normalized
    on any lag at all."""
    import math

    from morie.fn.ccf import ccf
    from morie.fn.ccfn import ccf_normalized

    x = [math.sin(i / 5.0) for i in range(40)]
    y = [math.cos(i / 5.0) for i in range(40)]
    r = ccf(x, y, nlags=8)
    assert len(r["ccf_values"]) == 17
    assert all(abs(v) <= 1.0 + 1e-12 for v in r["ccf_values"])
    n = ccf_normalized(x, y, maxlag=8)
    assert len(n.extra["ccf"]) == 9
