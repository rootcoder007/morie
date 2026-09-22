"""Tests for alfreq.alphafold_recycling."""

from morie.fn import _array_core as np

from morie.fn.alfreq import alphafold_recycling


def test_alfreq_basic():
    """Test basic functionality with documented argument shapes."""
    rng_z = np.random.default_rng(44)
    rng_w = np.random.default_rng(45)
    rng_m = np.random.default_rng(46)
    rng_x = np.random.default_rng(47)

    n = 5
    cm = 3
    cz = 4
    nb = 6

    z = rng_z.normal(0, 1, 100).tolist() if hasattr(rng_z.normal(0, 1, 100), "tolist") else rng_z.normal(0, 1, 100)
    # Build z with the documented n x n x cz shape.
    z = [[[float(v) for v in rng_z.normal(0, 1, cz)] for _ in range(n)] for _ in range(n)]

    # First MSA row m1: n x cm
    m1 = [[float(v) for v in rng_m.normal(0, 1, cm)] for _ in range(n)]

    # Predicted beta-carbon positions x: n x 3
    x = [[float(v) for v in rng_x.normal(0, 1, 3)] for _ in range(n)]

    # Projection wd: cz x len(bins)
    bins = [3.375 + i * 1.25 for i in range(nb)]
    wd = [[float(v) for v in rng_w.normal(0, 1, nb)] for _ in range(cz)]

    result = alphafold_recycling(m1, z, x, wd, bins=bins, ncycle=1)

    assert isinstance(result, dict)
    assert "z" in result
    assert "m1" in result
    assert "d" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result

    # n must equal the residue count.
    assert result["n"] == n

    # z and m1 have documented shapes n x n x cz and n x cm.
    assert len(result["z"]) == n
    assert all(len(result["z"][i]) == n for i in range(n))
    assert all(len(result["z"][i][j]) == cz for i in range(n) for j in range(n))

    assert len(result["m1"]) == n
    assert all(len(result["m1"][i]) == cm for i in range(n))

    # d is n x n.
    assert len(result["d"]) == n
    assert all(len(result["d"][i]) == n for i in range(n))


def test_alfreq_edge():
    """Test edge cases using the documented argument shapes."""
    rng_z = np.random.default_rng(44)
    rng_w = np.random.default_rng(45)
    rng_m = np.random.default_rng(46)
    rng_x = np.random.default_rng(47)

    n = 3
    cm = 2
    cz = 3
    nb = 4

    # Build z with the documented n x n x cz shape.
    z = [[[float(v) for v in rng_z.normal(0, 1, cz)] for _ in range(n)] for _ in range(n)]

    # First MSA row m1: n x cm
    m1 = [[float(v) for v in rng_m.normal(0, 1, cm)] for _ in range(n)]

    # Predicted beta-carbon positions x: n x 3
    x = [[float(v) for v in rng_x.normal(0, 1, 3)] for _ in range(n)]

    # Projection wd: cz x len(bins)
    bins = [2.0 + i for i in range(nb)]
    wd = [[float(v) for v in rng_w.normal(0, 1, nb)] for _ in range(cz)]

    # Zero ncycle should leave the initial state effectively unchanged for z
    # (no distance-based update is applied), while m1 is still layer-normed
    # on each iteration including zero iterations.
    result0 = alphafold_recycling(m1, z, x, wd, bins=bins, ncycle=0)
    assert isinstance(result0, dict)
    assert result0["n"] == n

    result1 = alphafold_recycling(m1, z, x, wd, bins=bins, ncycle=1)
    assert isinstance(result1, dict)
    assert result1["n"] == n

    result3 = alphafold_recycling(m1, z, x, wd, bins=bins, ncycle=3)
    assert isinstance(result3, dict)
    assert result3["n"] == n
