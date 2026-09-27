"""LAR lasso modification against lars(type = 'lasso') and forward stagewise against its definition and lars."""

import math

from morie.fn import esl_least_angle_reg
from morie.fn.eslfsw import esl_forward_stagewise
from morie.fn.esll1m import esl_l1_margin


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def data():
    i = range(1, 31)
    X = [[math.sin(t / 7), math.cos(t / 5), math.sin(t / 3 + 1), math.cos(t / 11) + 0.8 * math.sin(t / 7)] for t in i]
    return X, [r[0] * 1.5 - r[1] + 0.8 * r[3] + 0.3 * math.cos(t / 2) for r, t in zip(X, i)]


def test_lar_lasso_path_with_a_drop():
    X, y = data()
    a = esl_least_angle_reg(X, y, method="lasso")
    ref = [
        [0.0, 0.0, 0.0, 0.0],
        [2.588592218887, 0.0, 0.0, 0.0],
        [3.065356596316, -0.421867499939, 0.0, 0.0],
        [1.942484376524, -0.824381458848, 0.0, 0.550087740556],
        [0.0, -1.463389012870, -0.145090426698, 1.482803984280],
        [0.0, -1.473032148484, -0.158688503810, 1.490659239433],
        [-1.161959987102, -1.845632091993, -0.231880967910, 2.040738435453],
    ]
    got = a["coef_path"].tolist()
    assert len(got) == 7 and a["dropped"].tolist() == [0]  # lars actions 1 2 4 3 -1 1
    assert all(abs(u - v) < 1e-11 for gr, rr in zip(got, ref) for u, v in zip(gr, rr))
    plain = esl_least_angle_reg(X, y)
    assert plain["dropped"].tolist() == [] and plain["n_steps"] == 4


def test_forward_stagewise():
    X, y = data()
    f = esl_forward_stagewise(X, y, eps=0.001, max_steps=2000)
    P = f["path"]
    assert all(close(sum(abs(a - b) for a, b in zip(P[k + 1], P[k])), 0.001) for k in range(2000))
    xbar = [sum(r[j] for r in X) / 30 for j in range(4)]
    cols = [[r[j] - xbar[j] for r in X] for j in range(4)]
    cols = [[v / math.sqrt(sum(w * w for w in c)) for v in c] for c in cols]
    r = [v - sum(y) / 30 for v in y]
    for k in range(2000):
        cor = [sum(a * b for a, b in zip(c, r)) for c in cols]
        j = max(range(4), key=lambda q: abs(cor[q]))
        assert f["chosen"][k] == j
        d = P[k + 1][j] - P[k][j]
        assert close(d, 0.001 if cor[j] > 0 else -0.001)
        r = [ri - d * cj for ri, cj in zip(r, cols[j])]
    g = esl_forward_stagewise(X, [-v for v in y], eps=0.001, max_steps=2000)
    assert all(close(u, -v) for gr, fr in zip(g["path"], P) for u, v in zip(gr, fr))  # FS is odd in y
    m = esl_l1_margin([1, -1, 1], [0.8, -0.5, 0.2], [0.5, -0.3])
    assert close(m["margin"], 0.25) and m["argmin"] == 2
