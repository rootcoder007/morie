import math

import pytest

from morie.fn.research_sentence_bounds import contaminated_bounds, sentence_effect_bounds, sentence_effect_mtr

Y = [int((math.sin(2.3 * i) + 1) * 1.7) % 2 for i in range(53)]
Z = ["custody" if math.cos(1.1 * i) > -0.2 else "community" for i in range(53)]
W = [1 + (i % 7) / 3 for i in range(53)]


def _cells(y, z, w, lev):
    tw = math.fsum(w)
    joint = math.fsum(wi * yi for yi, zi, wi in zip(y, z, w) if zi == lev) / tw
    pz = math.fsum(wi for zi, wi in zip(z, w) if zi == lev) / tw
    return joint, pz


def test_worst_case_bounds_width_one_and_attained_ends():
    b = sentence_effect_bounds(Y, Z, W)
    assert b.levels == {"comparison": "community", "treatment": "custody"}
    for lev in ("community", "custody"):
        j, p = _cells(Y, Z, W, lev)
        tw = math.fsum(W)
        fill0 = math.fsum(w * (y if z == lev else 0) for y, z, w in zip(Y, Z, W)) / tw
        fill1 = math.fsum(w * (y if z == lev else 1) for y, z, w in zip(Y, Z, W)) / tw
        assert b.outcome_bounds[lev][0] == pytest.approx(fill0, abs=1e-12)
        assert b.outcome_bounds[lev][1] == pytest.approx(fill1, abs=1e-12)
        assert b.joint[lev] == pytest.approx(j, abs=1e-15) and b.pz[lev] == pytest.approx(p, abs=1e-15)
    assert b.ate_width == pytest.approx(1.0, abs=1e-12)
    assert b.ate_bounds["lower"] <= 0 <= b.ate_bounds["upper"]
    assert b.ate_bounds["lower"] <= b.naive_difference <= b.ate_bounds["upper"]
    jt, pt = _cells(Y, Z, W, "custody")
    jc, pc = _cells(Y, Z, W, "community")
    assert b.naive_difference == pytest.approx(jt / pt - jc / pc, abs=1e-12)
    flip = sentence_effect_bounds(Y, Z, W, contrast="community")
    assert flip.ate_bounds["lower"] == pytest.approx(-b.ate_bounds["upper"], abs=1e-12)
    with pytest.raises(ValueError, match="two distinct"):
        sentence_effect_bounds([0, 1], ["a", "a"])
    with pytest.raises(ValueError, match="0/1"):
        sentence_effect_bounds([0, 2], ["a", "b"])


def test_contaminated_bounds():
    qs = [0.02, 0.13, 0.5, 0.91, 1.0]
    p = 0.17
    b = contaminated_bounds(qs, p)
    assert b["lower"] == pytest.approx([max(0.0, (q - p) / (1 - p)) for q in qs], abs=1e-15)
    assert b["upper"] == pytest.approx([min(1.0, q / (1 - p)) for q in qs], abs=1e-15)
    assert b["width"] == pytest.approx([p / (1 - p)] * 5, abs=1e-15)
    assert b["informative"] == [(p < q) or (p < 1 - q) for q in qs]
    # a mixture with clean share cl always lies inside
    for cl, r in [(0.3, 0.9), (0.8, 0.0), (0.05, 1.0)]:
        q = (1 - p) * cl + p * r
        c = contaminated_bounds(q, p)
        assert c["lower"][0] - 1e-12 <= cl <= c["upper"][0] + 1e-12
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        contaminated_bounds(0.5, 1)


def test_mtr_bounds():
    m = sentence_effect_mtr(Y, Z, W)
    jt, pt = _cells(Y, Z, W, "custody")
    jc, pc = _cells(Y, Z, W, "community")
    assert m.bounds["lower"] == 0.0
    assert m.bounds["upper"] == pytest.approx(jt + (pc - jc), abs=1e-12)
    n = sentence_effect_mtr(Y, Z, W, direction="non-increasing")
    assert n.bounds["lower"] == pytest.approx(-((pt - jt) + jc), abs=1e-12) and n.bounds["upper"] == 0.0
    b = sentence_effect_bounds(Y, Z, W)
    assert b.ate_bounds["lower"] <= n.bounds["lower"] and m.bounds["upper"] <= b.ate_bounds["upper"] + 1e-12
