"""Tests for morie.fn.lsmextra (FRAGSTATS shape, core, proximity and information metrics)."""

import math

import pytest

from morie.fn.lsmextra import class_structure, landscape_information, patch_structure, proximity_metrics

L = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 2], [1, 1, 1, 2, 2], [2, 1, 1, 1, 1]]


def test_patch_structure_hand_values():
    r = patch_structure(L)
    # L-shaped class-2 patch (3 cells): 2 diagonal + 4 rook ordered pairs -> ((2 + 2 * 4 + 3) / 3 - 1) / 12
    assert r.contig[2] == pytest.approx(((2 + 8 + 3) / 3 - 1) / 12, abs=1e-15)
    assert r.contig[1] == 0.0
    # single cell: circle radius sqrt(2)/2, a = 1
    assert r.circle[1] == pytest.approx(1 - 1 / (math.pi * 0.5), abs=1e-15)
    # L patch encloses a 2x2 square of corners: radius sqrt(2)
    assert r.circle[2] == pytest.approx(1 - 3 / (math.pi * 2), abs=1e-12)
    cx, cy = 14 / 3, 8 / 3
    g = (math.hypot(5 - cx, 2 - cy) + math.hypot(4 - cx, 3 - cy) + math.hypot(5 - cx, 3 - cy)) / 3
    assert r.gyrate[2] == pytest.approx(g, abs=1e-15)
    assert [round(v * 1e4, 9) for v in r.core] == [3.0, 0.0, 0.0]
    assert [round(v * 1e4, 9) for v in patch_structure(L, consider_boundary=True).core] == [9.0, 0.0, 0.0]
    assert r.cai[0] == pytest.approx(100 * 3 / 16, abs=1e-12)
    assert r.frac[1] == 1.0
    sq = patch_structure([[1, 1], [1, 1]], res=10)
    assert sq.frac[0] == pytest.approx(2 * math.log(0.25 * 80) / math.log(400), abs=1e-15)


def test_circle_methods():
    diag = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ex = patch_structure(diag, directions=8)
    lm = patch_structure(diag, directions=8, circle_method="landscapemetrics")
    assert ex.circle[-1] >= lm.circle[-1]
    with pytest.raises(ValueError):
        patch_structure(diag, circle_method="bad")


def test_class_structure_means_and_pafrac():
    c = class_structure(L)
    p = patch_structure(L)
    assert c.contig_mn[1] == pytest.approx((p.contig[1] + p.contig[2]) / 2, abs=1e-15)
    assert c.gyrate_am[1] == pytest.approx(0.25 * p.gyrate[1] + 0.75 * p.gyrate[2], abs=1e-15)
    assert c.cpland[0] == pytest.approx(100 * 3 / 20, abs=1e-12)
    assert math.isnan(c.pafrac[0])


def test_proximity_similarity_connect():
    G = [[1, 0, 1], [0, 0, 0], [1, 0, 0]]
    r = proximity_metrics(G, search_radius=2, directions=4)
    assert r.prox == [0.0, 0.5, 0.25, 0.25]
    assert math.isnan(r.connect[0]) and r.connect[1] == pytest.approx(200 / 3)
    s = proximity_metrics(G, search_radius=2, directions=4, similarity=[[1, 0.5], [0.5, 1]])
    assert s.simi[0] == pytest.approx(3 * 0.5 * 1 / 1)
    assert s.simi[1] == pytest.approx(0.5 * 6 / 1 + 1 / 4 + 1 / 4)


def test_landscape_information():
    chk = [[(i + j) % 2 for j in range(6)] for i in range(6)]
    r = landscape_information(chk)
    assert r.ent == pytest.approx(1.0, abs=1e-15)
    assert r.condent == pytest.approx(0.0, abs=1e-15)
    assert r.mutinf == pytest.approx(1.0, abs=1e-15)
    q = landscape_information(L)
    assert q.condent == pytest.approx(q.joinent - q.ent, abs=1e-15)
    assert (round(q.ent, 6), round(q.condent, 6), round(q.mutinf, 6)) == (0.708836, 0.689275, 0.019561)
