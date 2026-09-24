"""Tests for prsclm.prs_cs_clump."""

from morie.fn import _array_core as np

from morie.fn.prsclm import prs_cs_clump


def _make_ld(m, rng):
    """Build an m x m symmetric non-negative matrix with unit diagonal."""
    A = rng.normal(0, 1, (m, m))
    R = [[abs(A[i][j]) for j in range(m)] for i in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            avg = (R[i][j] + R[j][i]) / 2.0
            R[i][j] = avg
            R[j][i] = avg
        R[i][i] = 1.0
    return R


def test_prsclm_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    m = 20

    beta = [float(v) for v in rng.normal(0, 0.1, m)]
    pv = [float(v) for v in rng.uniform(0, 1, m)]
    pos = [float(i) * 1000.0 for i in range(m)]
    sumstats = {"beta": beta, "p": pv, "position": pos}

    ld_ref = _make_ld(m, rng)

    result = prs_cs_clump(sumstats, ld_ref)
    assert isinstance(result, dict)
    assert "index_variants" in result
    assert "clump_of" in result
    assert len(result["clump_of"]) == m


def test_prsclm_edge():
    """Test edge case with a single variant."""
    sumstats = {"beta": [0.1], "p": [0.5], "position": [0.0]}
    ld_ref = [[1.0]]
    result = prs_cs_clump(sumstats, ld_ref, r2=1.0)
    assert isinstance(result, dict)
    assert "index_variants" in result
    assert "clump_of" in result
    assert len(result["clump_of"]) == 1
