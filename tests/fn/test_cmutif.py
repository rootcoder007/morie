"""Tests for cmutif.conditional_mi."""

import math

from morie.fn import _array_core as np

from morie.fn.cmutif import conditional_mi


def test_cmutif_basic():
    """Test basic functionality against the documented formula."""
    rng = np.random.default_rng(42)
    # 3-D joint pmf p[i][j][k]; non-negative, sums to 1.
    nx, ny, nz = 2, 3, 2
    pxyz = rng.random((nx, ny, nz))
    pxyz = pxyz / pxyz.sum()

    # Re-flatten into the nested-list shape the function expects.
    p_nested = [[[float(pxyz[i, j, k]) for k in range(nz)]
                 for j in range(ny)] for i in range(nx)]
    tot = sum(sum(sum(row) for row in plane) for plane in p_nested)
    p_norm = [[[v / tot for v in row] for row in plane] for plane in p_nested]

    def h(p):
        s = 0.0
        for v in p:
            if v > 0.0:
                s -= v * math.log(v, 2.0)
        return s

    def flat3(p):
        out = []
        for i in range(len(p)):
            for j in range(len(p[i])):
                for k in range(len(p[i][j])):
                    out.append(p[i][j][k])
        return out

    def marg3(p, axes):
        # axes are indices into the (i, j, k) tuple; remaining axes are summed out.
        idx = {0: 0, 1: 1, 2: 2}
        keep = [a for a in (0, 1, 2) if a in axes]
        out_shape = [len(p)] if 0 in keep else [1]
        if 1 in keep:
            out_shape.append(len(p[0]))
        else:
            out_shape.append(1)
        if 2 in keep:
            out_shape.append(len(p[0][0]))
        else:
            out_shape.append(1)
        # Build the marginal by iterating and bucketing.
        from collections import defaultdict
        bucket = defaultdict(float)
        for i in range(len(p)):
            for j in range(len(p[i])):
                for k in range(len(p[i][j])):
                    key = tuple([(i, j, k)[a] for a in keep])
                    bucket[key] += p[i][j][k]
        # Flatten bucket in lexicographic order of kept axes.
        sizes = [len(p), len(p[0]), len(p[0][0])]
        kept_sizes = [sizes[a] for a in keep]
        ordered = []
        from itertools import product
        for tup in product(*[range(s) for s in kept_sizes]):
            ordered.append(bucket[tup])
        return ordered

    hxyz = h(flat3(p_norm))
    hxz = h(marg3(p_norm, (0, 2)))
    hyz = h(marg3(p_norm, (1, 2)))
    hz = h(marg3(p_norm, (2,)))
    expected_estimate = hxz + hyz - hxyz - hz

    result = conditional_mi(pxyz)

    assert isinstance(result, dict)
    assert "estimate" in result
    for key in ("estimate", "hxz", "hyz", "hxyz", "hz", "n", "method"):
        assert key in result

    # Result.estimate must be a finite number for a valid joint pmf.
    assert math.isfinite(result["estimate"])

    # n must equal the total number of cells in the 3-D table.
    assert result["n"] == nx * ny * nz

    # The method string should mention "Conditional mutual information".
    assert isinstance(result["method"], str)
    assert "Conditional mutual information" in result["method"]

    # Independent recomputation from the documented formula must match.
    assert abs(result["estimate"] - expected_estimate) < 1e-10
    assert abs(result["hxyz"] - hxyz) < 1e-10
    assert abs(result["hxz"] - hxz) < 1e-10
    assert abs(result["hyz"] - hyz) < 1e-10
    assert abs(result["hz"] - hz) < 1e-10


def test_cmutif_edge():
    """Test edge cases: result keys and types for a valid 3-D joint pmf."""
    rng = np.random.default_rng(42)
    nx, ny, nz = 2, 2, 2
    pxyz = rng.random((nx, ny, nz))
    pxyz = pxyz / pxyz.sum()

    result = conditional_mi(pxyz)

    assert isinstance(result, dict)
    for key in ("estimate", "hxz", "hyz", "hxyz", "hz", "n", "method"):
        assert key in result
    assert math.isfinite(result["estimate"])
