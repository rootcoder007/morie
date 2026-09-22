"""Tests for alfsmd.alphafold_msa_attention."""

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.alfsmd import alphafold_msa_attention


def _build_inputs(s=2, n=3, cm=2, c=2, nh=1, cz=2, seed=42):
    """Build inputs that match the documented shapes for MSA attention.

    m  : s x n x cm
    wq, wk, wv, wg : nh lists, each a c x cm matrix
    wo : cm x (nh * c)
    z  : n x n x cz
    wb : nh x cz
    """
    rng = np.random.default_rng(seed)

    m = [[[float(rng.normal()) for _ in range(cm)] for _ in range(n)]
         for _ in range(s)]

    def _wm():
        return [[float(rng.normal()) for _ in range(cm)] for _ in range(c)]

    wq = [_wm() for _ in range(nh)]
    wk = [_wm() for _ in range(nh)]
    wv = [_wm() for _ in range(nh)]
    wg = [_wm() for _ in range(nh)]

    wo = [[float(rng.normal()) for _ in range(nh * c)]
          for _ in range(cm)]

    z = [[[float(rng.normal()) for _ in range(cz)] for _ in range(n)]
         for _ in range(n)]

    wb = [[float(rng.normal()) for _ in range(cz)] for _ in range(nh)]

    return m, wq, wk, wv, wg, wo, z, wb


def test_alfsmd_basic():
    """Test basic row-wise functionality with the full documented signature."""
    m, wq, wk, wv, wg, wo, z, wb = _build_inputs()
    result = alphafold_msa_attention(m, wq, wk, wv, wg, wo, z=z, wb=wb,
                                     mode="row")
    # The function returns a RichResult whose payload is a dict-like mapping.
    assert hasattr(result, "keys") or isinstance(result, dict)
    # The docstring lists these keys explicitly.
    for key in ("m", "attn", "estimate", "n", "s", "method"):
        assert key in result
    # Shapes propagated from inputs.
    s = len(m)
    n = len(m[0])
    nh = len(wq)
    cm = len(wo)
    assert result["n"] == n
    assert result["s"] == s
    # Each attention distribution must sum to one (documented property).
    for h in range(nh):
        for si in range(s):
            for i in range(n):
                row = result["attn"][h][si][i]
                assert abs(sum(row) - 1.0) < 1e-6
    # Output shape s x n x cm.
    assert len(result["m"]) == s
    assert len(result["m"][0]) == n
    assert len(result["m"][0][0]) == cm
    # Mode propagated.
    assert result["mode"] == "row"


def test_alfsmd_edge():
    """Test that mode='column' runs without the pair-representation inputs."""
    m, wq, wk, wv, wg, wo, _z, _wb = _build_inputs()
    result = alphafold_msa_attention(m, wq, wk, wv, wg, wo, mode="column")
    assert "m" in result
    assert "estimate" in result
    assert "attn" in result
    # Column-wise attention sums to one across the s axis.
    nh = len(wq)
    s = len(m)
    n = len(m[0])
    for h in range(nh):
        for si in range(s):
            for i in range(n):
                assert abs(sum(result["attn"][h][si][i]) - 1.0) < 1e-6
    assert result["mode"] == "column"
