"""Tests for nbeats. Full anchor: ledger/wave3/anchor_ts_family.py."""
import math
import pytest
from morie.fn import _s03core as k
from morie.fn.nbeats import (nbeats_forecast, nbeats_stack,
                             seasonality_basis, trend_basis)

SIG = [3.0 + 0.4 * t + 2.0 * math.sin(2 * math.pi * t / 12.0)
       for t in range(48)]


def test_the_bases_are_what_they_claim():
    tb = trend_basis(8, 2)
    assert len(tb) == 3
    assert tb[0] == [1.0] * 8
    assert tb[1][4] == pytest.approx(0.5)
    sb = seasonality_basis(12, 2)
    assert len(sb) == 4
    assert sb[0][0] == pytest.approx(1.0)
    assert sb[1][0] == pytest.approx(0.0, abs=1e-15)
    with pytest.raises(ValueError):
        trend_basis(8, -1)
    with pytest.raises(ValueError):
        seasonality_basis(8, 0)


def test_the_residual_telescopes_exactly():
    """x_L = x_0 - sum of backcasts. Skip the subtraction and every
    block re-fits the same trend."""
    blocks = [("trend", 2, 3), ("seasonality", 2, 3)]
    fc, resid, trace = nbeats_stack(SIG[:36], 12, blocks)
    for t in range(36):
        recon = SIG[t] - sum(b["backcast"][t] for b in trace)
        assert recon == pytest.approx(resid[t], abs=1e-10)
    assert trace[-1]["residual_norm"] < trace[0]["residual_norm"]


def test_it_forecasts_a_trend_plus_seasonal_signal():
    r = nbeats_forecast(SIG, 12, lookback=36)
    err = k.mean([abs(r["forecast"][h] - SIG[36 + h]) for h in range(12)])
    assert err < 0.25 * (max(SIG) - min(SIG))
    assert len(r["forecast"]) == 12


def test_argument_checks():
    with pytest.raises(ValueError):
        nbeats_stack(SIG[:36], 12, [("nope", 2, 3)])
    with pytest.raises(ValueError):
        nbeats_forecast(SIG, 12, lookback=2)
    with pytest.raises(ValueError):
        nbeats_stack(SIG[:36], 0, [("trend", 2, 3)])
