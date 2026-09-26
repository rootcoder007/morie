"""Tests for bsaclass.rangayyan_ica_artifact (Rangayyan sec. 9.7.2, 9.12)."""

import math

import pytest

from morie.fn.bsaclass import rangayyan_ica_artifact


N = 400
S1 = [math.sin(2 * math.pi * 7 * i / 100) for i in range(N)]
S2 = [5.0 if i % 97 == 0 else 0.0 for i in range(N)]
X = [[a + 0.8 * b for a, b in zip(S1, S2)], [0.6 * a + b for a, b in zip(S1, S2)],
     [0.3 * a - 0.5 * b + 0.1 * math.cos(0.3 * i) for i, (a, b) in enumerate(zip(S1, S2))]]


def _kurt_excess(v):
    m = sum(v) / len(v)
    s2 = sum((x - m) ** 2 for x in v) / len(v)
    return sum((x - m) ** 4 for x in v) / len(v) / s2 ** 2 - 3.0


def test_rgicaart_basic():
    """Components are ranked by kurtosis excess K - 3 (eq. 3.5): a
    sinusoid has -1.5, the sparse blink train is strongly positive and is
    the only component flagged; zeroing it and back-projecting leaves the
    spikes out of every channel."""
    r = rangayyan_ica_artifact(X)
    k = r["kurtosis"]
    for comp, kk in zip(r["components"], k):
        assert kk == pytest.approx(_kurt_excess(comp), rel=1e-9)
    assert r["artifacts"] == [max(range(len(k)), key=lambda i: k[i])]
    assert k[r["artifacts"][0]] > 20
    spikes = [i for i in range(N) if i % 97 == 0]
    for ch_clean, ch_raw in zip(r["clean"], X):
        assert max(abs(ch_clean[i]) for i in spikes) < 0.5 * max(abs(ch_raw[i]) for i in spikes)


def test_rgicaart_edge():
    """Fewer than four samples per channel cannot be separated."""
    with pytest.raises(ValueError):
        rangayyan_ica_artifact([[0.1, 0.2, 0.3], [0.3, 0.1, 0.2]])
