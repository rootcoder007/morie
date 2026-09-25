"""Tests for volgkr.vol_garman_klass."""

import math

import pytest

from morie.fn.volgkr import vol_garman_klass

# six bars with low <= open, close <= high throughout
O = [100.0, 101.0, 102.0, 99.0, 98.0, 100.0]
H = [102.0, 103.0, 103.0, 101.0, 100.0, 104.0]
L = [99.0, 100.0, 98.0, 97.0, 96.0, 99.0]
C = [101.0, 102.0, 99.0, 98.0, 100.0, 103.0]

_K = 2.0 * math.log(2.0) - 1.0


def test_volgkr_matches_equation_20():
    out = vol_garman_klass(O, H, L, C)
    hl = [math.log(h / l) ** 2 for h, l in zip(H, L)]
    co = [math.log(c / o) ** 2 for c, o in zip(C, O)]
    per_bar = [0.5 * a - _K * b for a, b in zip(hl, co)]
    var = sum(per_bar) / len(per_bar)
    assert out["variance"] == pytest.approx(var, rel=1e-12)
    assert out["sigma"] == pytest.approx(math.sqrt(var), rel=1e-12)
    assert out["range_term"] == pytest.approx(sum(0.5 * a for a in hl) / 6, rel=1e-12)
    assert out["openclose_term"] == pytest.approx(sum(_K * b for b in co) / 6, rel=1e-12)
    # the open-close term is SUBTRACTED, so the two terms bracket the variance
    assert out["variance"] == pytest.approx(
        out["range_term"] - out["openclose_term"], rel=1e-12)
    assert out["openclose_term"] > 0.0
    assert out["variance"] < out["range_term"]
    assert out["n"] == 6
    assert out["efficiency_vs_close"] == 7.4
    assert out["sigma_annualised"] is None


def test_volgkr_annualises_by_the_square_root_of_the_period_count():
    out = vol_garman_klass(O, H, L, C, periods_per_year=252)
    plain = vol_garman_klass(O, H, L, C)
    assert out["variance"] == pytest.approx(plain["variance"], rel=1e-12)
    assert out["sigma_annualised"] == pytest.approx(
        plain["sigma"] * math.sqrt(252.0), rel=1e-12)


def test_volgkr_reduces_to_the_range_term_when_every_bar_closes_where_it_opened():
    """C == O kills the trend correction exactly, leaving half the mean
    squared log range -- twice Parkinson's 1/(4 log 2) coefficient."""
    o = [100.0, 100.0, 100.0, 100.0]
    h = [102.0, 101.0, 104.0, 103.0]
    lo = [99.0, 98.0, 97.0, 96.0]
    out = vol_garman_klass(o, h, lo, list(o))
    want = sum(0.5 * math.log(a / b) ** 2 for a, b in zip(h, lo)) / 4
    assert out["openclose_term"] == pytest.approx(0.0, abs=1e-15)
    assert out["variance"] == pytest.approx(want, rel=1e-12)
    assert out["range_term"] == pytest.approx(want, rel=1e-12)


def test_volgkr_no_bar_can_be_negative_once_the_ohlc_ordering_holds():
    """|log C/O| <= log H/L whenever low <= open, close <= high, and
    0.5 > 2 log 2 - 1, so every per-bar value is non-negative. The
    extreme bar -- open at the low, close at the high -- is the tightest
    case and still leaves 0.5 - (2 log 2 - 1) of the squared range."""
    r = math.log(104.0 / 100.0)
    out = vol_garman_klass([100.0, 100.0], [104.0, 104.0], [100.0, 100.0],
                           [104.0, 104.0])
    assert out["negative_bar_fraction"] == 0.0
    assert out["variance"] == pytest.approx((0.5 - _K) * r * r, rel=1e-12)
    assert vol_garman_klass(O, H, L, C)["negative_bar_fraction"] == 0.0


def test_volgkr_rejects_bad_input():
    with pytest.raises(ValueError, match="share a length"):
        vol_garman_klass(O, H, L, C[:-1])
    with pytest.raises(ValueError, match="at least 2 bars"):
        vol_garman_klass([100.0], [101.0], [99.0], [100.0])
    with pytest.raises(ValueError, match="positive"):
        vol_garman_klass([100.0, 100.0], [101.0, 101.0], [0.0, 99.0], [100.0, 100.0])
    with pytest.raises(ValueError, match="low <= open"):
        vol_garman_klass([100.0, 100.0], [101.0, 101.0], [99.0, 99.0],
                         [102.0, 100.0])
    # a flat tape has zero Garman-Klass variance, which the estimator
    # refuses rather than returning as a volatility
    with pytest.raises(ValueError, match="not positive"):
        vol_garman_klass([100.0] * 5, [100.0] * 5, [100.0] * 5, [100.0] * 5)
