"""Tests for tritta.alphafold_triangle_attn (AF2 Supplementary Algorithms 13-14)."""

import math

import pytest

from morie.fn.tritta import alphafold_triangle_attn


def _w(r, c, s):
    return [[math.sin(s + 1.7 * i + 0.9 * j) * 0.5 for j in range(c)] for i in range(r)]


def _setup(n=3, cz=4, c=2, nh=2):
    z = [[[math.cos(0.3 * i + 0.7 * j + 1.1 * t) for t in range(cz)] for j in range(n)] for i in range(n)]
    wq = [_w(c, cz, 1 + h) for h in range(nh)]
    wk = [_w(c, cz, 5 + h) for h in range(nh)]
    wv = [_w(c, cz, 9 + h) for h in range(nh)]
    wg = [_w(c, cz, 13 + h) for h in range(nh)]
    wb = [[math.sin(17 + h + 0.4 * t) for t in range(cz)] for h in range(nh)]
    wo = _w(cz, nh * c, 21)
    return z, wq, wk, wv, wb, wg, wo


def _ref(z, wq, wk, wv, wb, wg, wo, mode):
    """Line by line: LayerNorm (eps 1e-5, no affine); q, k, v, g per head;
    b^h_ij = w_b^h . z_ij; starting node: a_ijk = softmax_k(q_ij.k_ik/sqrt c
    + b_jk), o_ij = g_ij * sum_k a_ijk v_ik; ending node: a_ijk =
    softmax_k(q_ij.k_kj/sqrt c + b_ki), o_ij = g_ij * sum_k a_ijk v_kj;
    output W_o concat_h(o^h_ij)."""
    n, nh, c = len(z), len(wq), len(wq[0])

    def ln(v):
        m = sum(v) / len(v)
        s = math.sqrt(sum((x - m) ** 2 for x in v) / len(v) + 1e-5)
        return [(x - m) / s for x in v]

    def mv(W, v):
        return [sum(a * b for a, b in zip(row, v)) for row in W]

    zn = [[ln(z[i][j]) for j in range(n)] for i in range(n)]
    out = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cat = []
            for h in range(nh):
                q = mv(wq[h], zn[i][j])
                g = [1 / (1 + math.exp(-x)) for x in mv(wg[h], zn[i][j])]
                if mode == "starting":
                    lg = [sum(a * b for a, b in zip(q, mv(wk[h], zn[i][k]))) / math.sqrt(c)
                          + sum(a * b for a, b in zip(wb[h], zn[j][k])) for k in range(n)]
                    vs = [mv(wv[h], zn[i][k]) for k in range(n)]
                else:
                    lg = [sum(a * b for a, b in zip(q, mv(wk[h], zn[k][j]))) / math.sqrt(c)
                          + sum(a * b for a, b in zip(wb[h], zn[k][i])) for k in range(n)]
                    vs = [mv(wv[h], zn[k][j]) for k in range(n)]
                mx = max(lg)
                e = [math.exp(x - mx) for x in lg]
                a = [x / sum(e) for x in e]
                cat += [g[t] * sum(a[k] * vs[k][t] for k in range(n)) for t in range(c)]
            out[i][j] = mv(wo, cat)
    return out


@pytest.mark.parametrize("mode", ["starting", "ending"])
def test_tritta_basic(mode):
    """Every output entry equals the line-by-line recomputation, and
    every attention distribution sums to one."""
    args = _setup()
    r = alphafold_triangle_attn(*args, mode=mode)
    ref = _ref(*args, mode)
    for i in range(3):
        for j in range(3):
            assert r["z"][i][j] == pytest.approx(ref[i][j], abs=1e-12)
            for h in range(2):
                assert sum(r["attn"][h][i][j]) == pytest.approx(1.0, abs=1e-14)


def test_tritta_edge():
    """An unknown mode raises."""
    with pytest.raises(ValueError):
        alphafold_triangle_attn(*_setup(), mode="middle")
