"""Tests for hmyolo.geron_yolo (grid decoding plus non-max suppression)."""

import pytest

from morie.fn import _array_core as np
from morie.fn.hmyolo import geron_yolo

S = 3


def _decode(i, j, tx, ty, w, h):
    cx, cy = (j + tx) / S, (i + ty) / S
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)


CELLS = [  # (i, j, tx, ty, w, h, conf, class)
    (0, 0, 0.5, 0.5, 0.4, 0.4, 0.9, 0),
    (0, 1, 0.1, 0.5, 0.4, 0.4, 0.7, 0),   # overlaps the first, same class
    (2, 2, 0.3, 0.6, 0.2, 0.3, 0.8, 1),
    (1, 0, 0.5, 0.5, 0.2, 0.2, 0.3, 1),   # below the confidence threshold
]


def _model(x):
    p = np.zeros((S, S, 7))
    for i, j, tx, ty, w, h, c, k in CELLS:
        p[i, j, :5] = [tx, ty, w, h, c]
        p[i, j, 5 + k] = 1.0
    return p


def test_hmyolo_basic():
    """Boxes decoded as ((j + tx)/S, (i + ty)/S, w, h); the lower-scoring
    same-class box is suppressed when its IoU with the kept one, computed
    here, exceeds the threshold."""
    b0, b1 = _decode(*CELLS[0][:6]), _decode(*CELLS[1][:6])
    iou = _iou(b0, b1)
    assert iou > 0.3
    result = geron_yolo(None, _model, conf_threshold=0.5, iou_threshold=0.3)
    assert isinstance(result, dict)
    assert int(result["n_detections"]) == 2
    assert int(result["n_candidates"]) == 3
    kept = sorted(zip([float(s) for s in result["scores"]],
                      [int(c) for c in result["classes"]],
                      [tuple(float(v) for v in b) for b in result["boxes"]]), reverse=True)
    assert kept[0][:2] == (0.9, 0) and kept[0][2] == pytest.approx(b0, abs=1e-12)
    assert kept[1][:2] == (0.8, 1)
    assert kept[1][2] == pytest.approx(_decode(*CELLS[2][:6]), abs=1e-12)


def test_hmyolo_edge():
    """Raising the IoU threshold above the pair's overlap keeps both."""
    iou = _iou(_decode(*CELLS[0][:6]), _decode(*CELLS[1][:6]))
    r = geron_yolo(None, _model, conf_threshold=0.5, iou_threshold=min(1.0, iou + 0.05))
    assert int(r["n_detections"]) == 3
    with pytest.raises(ValueError):
        geron_yolo(None, lambda x: np.zeros((2, 2, 3)))


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.hmyolo as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
