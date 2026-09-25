"""Tests for sam2vd.sam2_video_propagation (SAM 2 streaming memory, Ravi et al. 2024)."""

import pytest

from morie.fn.sam2vd import memory_bank, push_memory, sam2_video_propagation

FRAMES = [[float(t), 1.0 - 0.1 * t, 0.5] for t in range(6)]


def _enc(f):
    return list(f)


def _dec(feat, prompt):
    # "mask" = rounded features, plus the prompt when there is one
    return [round(v, 12) for v in feat] + ([prompt] if prompt is not None else [])


def test_sam2vd_basic():
    """Frame 0 meets an empty memory, so it is decoded from its own
    encoding untouched (SAM 2 is SAM on one image); later frames are
    conditioned on the bank; prompts reach the decoder on their frames."""
    r = sam2_video_propagation(FRAMES, _enc, _dec, prompts={0: "click", 3: "box"})
    assert r["n_frames"] == 6
    assert r["masks"][0] == [round(v, 12) for v in FRAMES[0]] + ["click"]
    assert r["conditioned"] == [False] + [True] * 5
    assert r["masks"][3][-1] == "box"


def test_sam2vd_edge():
    """The recent queue is FIFO with capacity n_recent; a prompted
    memory lives in its own queue and recent traffic never evicts it."""
    b = memory_bank(n_recent=2, m_prompted=1)
    b = push_memory(b, 0, [1.0], prompted=True)
    for t in range(1, 6):
        b = push_memory(b, t, [float(t)])
    assert [e["frame"] for e in b["recent"]] == [4, 5]
    assert [e["frame"] for e in b["prompted"]] == [0]
    with pytest.raises(ValueError):
        memory_bank(0, 1)
