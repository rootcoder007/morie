"""Tests for alfipa.alphafold_invariant_point."""

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.alfipa import alphafold_invariant_point


def _make_frames(n, seed=42):
    """Construct n identity frames [R, t] with R = I, t = 0."""
    return [[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
             [0.0, 0.0, 0.0]] for _ in range(n)]


def _matvec(w, x):
    """Apply row-major weight matrix w to vector x."""
    return [sum(w[r][c] * x[c] for c in range(len(x))) for r in range(len(w))]


def _vdot(a, b):
    return sum(a[t] * b[t] for t in range(len(a)))


def _vsub(a, b):
    return [a[t] - b[t] for t in range(len(a))]


def _vnorm2(a):
    return sum(x * x for x in a)


def _smax(logits):
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _apply_identity(R, t, p):
    return [t[i] + sum(R[i][k] * p[k] for k in range(3)) for i in range(3)]


def test_alfipa_basic():
    """Test basic functionality with n=2 residues, nh=2 heads, cs=cz=c=1, nqp=npv=1.

    With identity frames and n=1 (a single residue pair), the algorithm
    degenerates to a simple softmax over one element per head, so the
    attention weights are trivially [1.0] and the result is computable
    by hand.
    """
    n = 1
    cs = 1
    cz = 1
    c = 1
    nh = 1
    nqp = 1
    npv = 1

    # Single-residue single representation (n=1, cs=1).
    s = [[1.0]]

    # Pair representation (n x n x cz) = (1 x 1 x 1).
    z = [[[2.0]]]

    frames = _make_frames(n)

    # Per-head scalar projections wq, wk, wv: each nh x c x cs.
    wq = [[[1.0]]]
    wk = [[[1.0]]]
    wv = [[[1.0]]]

    # Per-head per-point projections wqp, wkp: nh x nqp x 3 x cs.
    wqp = [[[[1.0], [0.0], [0.0]]]]
    wkp = [[[[1.0], [0.0], [0.0]]]]

    # Per-head per-point value projections wvp: nh x npv x 3 x cs.
    wvp = [[[[1.0], [0.0], [0.0]]]]

    # Per-head pair-bias projection wb: nh x cz.
    wb = [[0.5]]

    # Per-head scalar weighting the point term.
    gamma = [0.0]

    # Output projection wo: cs x (nh * (cz + c + 4 * npv)).
    # Here nh=1, cz=1, c=1, npv=1, so width = 1 + 1 + 4 = 6.
    wo = [[0.0] * 6]  # zero output -> s update is zero

    result = alphafold_invariant_point(
        s, z, frames, wq, wk, wv, wqp, wkp, wvp, wb, gamma, wo
    )

    # Must be a RichResult with the documented keys.
    assert hasattr(result, "keys") or isinstance(result, dict)
    if isinstance(result, dict):
        keys = result.keys()
    else:
        keys = result.keys()

    for k in ("s", "attn", "points", "estimate", "n", "method"):
        assert k in keys, f"missing key {k!r}"

    # n = number of residues.
    assert result["n"] == n

    # With one residue, the softmax over a single logit is [1.0].
    assert len(result["attn"]) == nh
    assert len(result["attn"][0]) == n
    for h in range(nh):
        for i in range(n):
            assert len(result["attn"][h][i]) == n
            for j in range(n):
                assert abs(result["attn"][h][i][j] - 1.0) < 1e-9

    # With wo = 0, the update s is the zero vector of length cs.
    assert len(result["s"]) == n
    assert len(result["s"][0]) == cs
    for i in range(n):
        for t in range(cs):
            assert abs(result["s"][i][t]) < 1e-9

    # method is a string.
    assert isinstance(result["method"], str)


def test_alfipa_edge():
    """Test edge cases with n=2 residues and nh=1."""
    n = 2
    cs = 1
    cz = 1
    c = 1
    nh = 1
    nqp = 1
    npv = 1

    s = [[0.5], [-0.5]]
    z = [[[0.1], [0.2]], [[0.3], [0.4]]]
    frames = _make_frames(n)

    wq = [[[1.0]]]
    wk = [[[1.0]]]
    wv = [[[1.0]]]
    wqp = [[[[1.0], [0.0], [0.0]]]]
    wkp = [[[[1.0], [0.0], [0.0]]]]
    wvp = [[[[1.0], [0.0], [0.0]]]]
    wb = [[0.0]]
    gamma = [0.0]

    # Output width = nh*(cz + c + 4*npv) = 1*(1+1+4) = 6.
    wo = [[0.0] * 6]

    result = alphafold_invariant_point(
        s, z, frames, wq, wk, wv, wqp, wkp, wvp, wb, gamma, wo
    )

    # Documented keys present.
    keys = result.keys()
    for k in ("s", "attn", "points", "estimate", "n", "method"):
        assert k in keys, f"missing key {k!r}"

    assert result["n"] == n

    # With zero pair bias, gamma=0, and the q/k point contributions being
    # identical across residues (same projection, same frame), the logit
    # for (i, j) is wL * scale * q_i * k_j with wL = sqrt(1/3),
    # scale = 1/sqrt(c) = 1, q_i = s_i, k_j = s_j.
    scale = 1.0 / math.sqrt(c)
    wL = math.sqrt(1.0 / 3.0)

    # Independent computation of attention weights for head 0.
    logits = [[wL * scale * s[i][0] * s[j][0] for j in range(n)]
              for i in range(n)]
    # Softmax along each row independently.
    expected_attn = []
    for i in range(n):
        m = max(logits[i])
        exps = [math.exp(l - m) for l in logits[i]]
        ssum = sum(exps)
        expected_attn.append([e / ssum for e in exps])

    for i in range(n):
        for j in range(n):
            assert abs(result["attn"][0][i][j] - expected_attn[i][j]) < 1e-9

    # Output is zero because wo is zero.
    assert len(result["s"]) == n
    for i in range(n):
        for t in range(cs):
            assert abs(result["s"][i][t]) < 1e-9

    # estimate is a scalar float (mean of flattened s).
    assert isinstance(result["estimate"], float)
