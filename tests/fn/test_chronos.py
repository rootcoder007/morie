"""Tests for chronos.chronos_foundation_ts (Chronos tokenisation,
Ansari et al. 2024 Sec. 3.1): mean scaling, then uniform quantisation."""

import math

from morie.fn import _array_core as np
from morie.fn.chronos import EOS, PAD, chronos_foundation_ts, uniform_bins


def test_chronos_basic():
    """Tokens recomputed from the definitions: s = mean|x| over the
    context, token = number of bin edges at or below x/s, EOS appended."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    bins = uniform_bins(-15.0, 15.0, 4096)
    result = chronos_foundation_ts(y, bins)
    assert isinstance(result, dict)
    v = [float(q) for q in y.tolist()]
    s = math.fsum(abs(q) for q in v) / len(v)
    assert abs(result["scale"] - s) <= 1e-12 * s
    expect = [sum(1 for e in bins["edges"] if q / s >= e) for q in v]
    assert result["tokens"] == expect + [EOS]
    assert result["estimate"] == result["tokens"]
    assert result["n_clipped"] == 0
    assert result["vocab_size"] == 4096 + 2


def test_chronos_edge():
    """Clipping outside [c_1, c_B] and left padding: with 3 bins at
    -1, 0, 1 the series [1, -3, 2] has s = 2, scaled [0.5, -1.5, 1];
    -1.5 < c_1 is clipped to token 0, 0.5 and 1 map to token 2."""
    bins = uniform_bins(-1.0, 1.0, 3)
    result = chronos_foundation_ts([1.0, -3.0, 2.0], bins, pad_to=6)
    assert result["scale"] == 2.0
    assert result["n_clipped"] == 1
    assert result["tokens"] == [PAD, PAD, 2, 0, 2, EOS]
