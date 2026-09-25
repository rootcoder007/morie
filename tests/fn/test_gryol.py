"""Tests for gryol.geron_yolo_grid_loss (YOLOv1 sum-squared loss)."""

import math

import pytest

from morie.fn.gryol import geron_yolo_grid_loss


def _grid():
    # 2x2 grid, 2 classes; cells (0,0) and (1,1) hold objects
    T = [[[0.4, 0.6, 0.25, 0.36, 1.0, 1.0, 0.0], [0.0] * 7],
         [[0.0] * 7, [0.5, 0.2, 0.49, 0.16, 1.0, 0.0, 1.0]]]
    P = [[[0.5, 0.5, 0.16, 0.25, 0.8, 0.7, 0.3], [0.2, 0.1, 0.3, 0.3, 0.3, 0.5, 0.5]],
         [[0.9, 0.9, 0.1, 0.1, 0.1, 0.2, 0.8], [0.45, 0.3, 0.64, 0.09, 0.6, 0.1, 0.9]]]
    return P, T


def test_gryol_basic():
    """The loss recomputed term by term from the YOLOv1 definition:
    coordinates (square-rooted w, h) and class errors only in object
    cells, objectness in object cells, lam_noobj * conf^2 elsewhere."""
    P, T = _grid()
    result = geron_yolo_grid_loss(P, T)
    assert isinstance(result, dict)
    coord = obj = noobj = cls = 0.0
    for i in range(2):
        for j in range(2):
            p, t = P[i][j], T[i][j]
            if t[4] == 1.0:
                coord += 5.0 * ((p[0] - t[0]) ** 2 + (p[1] - t[1]) ** 2
                                + (math.sqrt(p[2]) - math.sqrt(t[2])) ** 2
                                + (math.sqrt(p[3]) - math.sqrt(t[3])) ** 2)
                obj += (p[4] - t[4]) ** 2
                cls += sum((p[5 + c] - t[5 + c]) ** 2 for c in range(2))
            else:
                noobj += 0.5 * (p[4] - t[4]) ** 2
    assert result["loss_coord"] == pytest.approx(coord, rel=1e-14)
    assert result["loss_obj"] == pytest.approx(obj, rel=1e-14)
    assert result["loss_noobj"] == pytest.approx(noobj, rel=1e-14)
    assert result["loss_class"] == pytest.approx(cls, rel=1e-14)
    assert result["loss"] == pytest.approx(coord + obj + noobj + cls, rel=1e-14)
    assert result["n_objects"] == 2


def test_gryol_edge():
    """Setting both lambdas to zero leaves only objectness and class
    error; mismatched shapes are refused."""
    P, T = _grid()
    r = geron_yolo_grid_loss(P, T, lam_coord=0.0, lam_noobj=0.0)
    full = geron_yolo_grid_loss(P, T)
    assert r["loss"] == pytest.approx(full["loss_obj"] + full["loss_class"], rel=1e-14)
    with pytest.raises(ValueError):
        geron_yolo_grid_loss(P, [row[:1] for row in T])


# --- appended: the module's own worked example as a gate -----------

import doctest as _doctest

import morie.fn.gryol as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
