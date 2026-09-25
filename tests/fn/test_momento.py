"""Tests for momento.moment_foundation."""

import math

import pytest

from morie.fn.momento import mask_patches, masked_loss, moment_foundation, task_mask


def test_momento_basic():
    """Each channel is truncated to whole patches and standardised on its
    own (ddof = 1); the batch shares the smallest patch count."""
    a = [[float(t), 10.0 * t] for t in range(10)]      # 2 channels, 10 points
    b = [[2.0 * t * t] for t in range(7)]               # 1 channel, 7 points
    r = moment_foundation([a, b], patch_len=3)
    assert r["n_series"] == 3 and r["patch_len"] == 3
    assert r["n_patches"] == 2
    assert [m["n_patches"] for m in r["meta"]] == [3, 3, 2]
    col = [float(t) for t in range(9)]
    m = sum(col) / 9
    sd = math.sqrt(sum((v - m) ** 2 for v in col) / 8)
    assert r["meta"][0]["mean"] == pytest.approx(m, rel=1e-15)
    assert r["batch"][0] == [[(v - m) / sd for v in col[0:3]], [(v - m) / sd for v in col[3:6]]]
    # the second channel is 10x the first, so it standardises identically
    flat = [[v for p in r["batch"][j] for v in p] for j in (0, 1)]
    assert flat[1] == pytest.approx(flat[0], rel=1e-12)


def test_momento_edge():
    """The loss scores masked positions only; masks follow the task."""
    P = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    mk = mask_patches(P, [1])
    assert mk["masked"] == [[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]]
    assert mk["mask"] == [False, True, False]
    rec = [[9.0, 9.0], [3.5, 3.0], [9.0, 9.0]]
    L = masked_loss(P, rec, mk["mask"])
    assert L["mse"] == pytest.approx((0.25 + 1.0) / 2, rel=1e-15) and L["n_scored"] == 2
    assert task_mask(8, "forecast", span=3) == [5, 6, 7]
    assert task_mask(8, "impute", span=2) == [3, 4]
    with pytest.raises(ValueError, match="every patch"):
        mask_patches(P, [0, 1, 2])
    with pytest.raises(ValueError, match="fewer"):
        moment_foundation([[[1.0], [2.0]]], patch_len=3)


