"""Tests for blockMx.block_maxima."""
from morie.fn.blockMx import block_maxima
from morie.fn.evgevs import evt_gev_sample


def test_recovers_gev_from_blocked_stream():
    # build a stream whose 50-block maxima are themselves GEV draws
    import math
    xs = evt_gev_sample(400, 8.0, 1.5, 0.1, seed=11)["x"]
    stream = []
    for m in xs:
        stream += [m - 5.0] * 49 + [m]
    r = block_maxima(stream, 50)
    assert abs(r["mu"] - 8.0) < 0.5
    assert r["n_blocks"] == 400
