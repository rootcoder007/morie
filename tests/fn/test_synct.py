"""Tests for synct.synthetic_control (Abadie-Diamond-Hainmueller)."""

import itertools
import math

import pytest

from morie.fn.synct import synthetic_control


def _panel():
    f = [math.sin(0.9 * t) + 0.1 * t for t in range(10)]
    load = [1.0, 0.6, 1.4, 0.3, 1.1]
    Y = [[a * v + 0.02 * math.cos(3.1 * t + u) for t, v in enumerate(f)] for u, a in enumerate(load)]
    for t in range(6, 10):
        Y[0][t] += 2.0
    return Y


def _simplex_ls(A, b):
    """min |A w - b|^2 over the simplex, exactly: for every support set S
    solve the equality-constrained least squares (KKT system with one
    multiplier) and keep the feasible solution with the smallest loss."""
    m = len(A[0])
    best, bw = float("inf"), None
    for k in range(1, m + 1):
        for S in itertools.combinations(range(m), k):
            n = k + 1
            M = [[0.0] * (n + 1) for _ in range(n)]
            for a, i in enumerate(S):
                for c, j in enumerate(S):
                    M[a][c] = 2 * sum(A[t][i] * A[t][j] for t in range(len(b)))
                M[a][k] = 1.0
                M[a][n] = 2 * sum(A[t][i] * b[t] for t in range(len(b)))
                M[k][a] = 1.0
            M[k][n] = 1.0
            ok = True
            for c in range(n):
                p = max(range(c, n), key=lambda r: abs(M[r][c]))
                if abs(M[p][c]) < 1e-14:
                    ok = False
                    break
                M[c], M[p] = M[p], M[c]
                for r in range(n):
                    if r != c:
                        fct = M[r][c] / M[c][c]
                        M[r] = [x - fct * y for x, y in zip(M[r], M[c])]
            if not ok:
                continue
            ws = [M[a][n] / M[a][a] for a in range(k)]
            if min(ws) < -1e-12:
                continue
            w = [0.0] * m
            for a, i in enumerate(S):
                w[i] = ws[a]
            loss = sum((sum(A[t][j] * w[j] for j in range(m)) - b[t]) ** 2 for t in range(len(b)))
            if loss < best - 1e-15:
                best, bw = loss, w
    return bw


def _fit(Y, row, t0, exclude=()):
    donors = [u for u in range(len(Y)) if u != row and u not in exclude]
    A = [[Y[u][t] for u in donors] for t in range(t0)]
    w = _simplex_ls(A, Y[row][:t0])
    synth = [sum(w[a] * Y[u][t] for a, u in enumerate(donors)) for t in range(len(Y[0]))]
    return w, [Y[row][t] - synth[t] for t in range(len(Y[0]))]


def _rmspe(g, a, b):
    return math.sqrt(sum(v * v for v in g[a:b]) / (b - a))


def test_synct_basic():
    """Weights are the simplex-constrained least-squares fit on the
    pre-period (solved exactly by enumerating supports); the estimate is
    the mean post gap; the placebo p-value ranks the treated unit's
    post/pre RMSPE ratio among placebos fitted WITHOUT the treated unit
    in their donor pools."""
    Y = _panel()
    w, gap = _fit(Y, 0, 6)
    r = synthetic_control(Y, None, None, 0, 6)
    # 1e-8: simplex_lstsq stops when its iterate moves less than 1e-12,
    # and the nearly collinear pre-period donor paths (condition ~1e3)
    # leave the weights about 2e-9 from the exact vertex solution
    assert [float(v) for v in r["weights"]] == pytest.approx(w, abs=1e-8)
    assert r["estimate"] == pytest.approx(sum(gap[6:]) / 4, abs=1e-8)
    ratio = _rmspe(gap, 6, 10) / _rmspe(gap, 0, 6)
    pr = []
    for j in range(1, 5):
        _, gj = _fit(Y, j, 6, exclude=(0,))
        pr.append(_rmspe(gj, 6, 10) / _rmspe(gj, 0, 6))
    assert r["placebo_p"] == pytest.approx((1 + sum(1 for q in pr if q >= ratio)) / 5, abs=1e-15)
    assert r["p_value_floor"] == pytest.approx(0.2, abs=1e-15)
    assert r["estimate"] == pytest.approx(2.0, abs=0.1)


def test_synct_edge():
    """Two units, an unknown treated unit, and fewer than two
    pre-periods raise."""
    Y = _panel()
    with pytest.raises(ValueError):
        synthetic_control(Y[:2], None, None, 0, 6)
    with pytest.raises(ValueError):
        synthetic_control(Y, None, None, 9, 6)
    with pytest.raises(ValueError):
        synthetic_control(Y, None, None, 0, 1)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.synct as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
