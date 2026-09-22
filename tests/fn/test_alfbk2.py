"""Tests for alfbk2.alphafold_backbone."""

import math

from morie.fn import _array_core as np

from morie.fn.alfbk2 import alphafold_backbone


def test_alfbk2_basic():
    """Test basic functionality."""
    rng = np.random.default_rng(42)
    n, cs = 4, 5
    s = [rng.normal(0, 1, cs).tolist() for _ in range(n)]
    w = [rng.normal(0, 1, cs).tolist() for _ in range(6)]

    result = alphafold_backbone(s, w)

    assert isinstance(result, dict)
    assert "frames" in result
    assert "quat" in result
    assert "estimate" in result
    assert "n" in result
    assert "method" in result

    assert result["n"] == n
    assert len(result["frames"]) == n
    assert len(result["quat"]) == n

    # Independent computation of the estimate from the documented formula:
    # quat normalisation fixes the leading component to 1 before dividing by
    # sqrt(1 + qx^2 + qy^2 + qz^2).  Estimate is mean of all translation
    # components across residues.
    est_sum = 0.0
    count = 0
    for i in range(n):
        p = [sum(w[r][c] * s[i][c] for c in range(cs)) for r in range(6)]
        nq = math.sqrt(1.0 + p[0] * p[0] + p[1] * p[1] + p[2] * p[2])
        est_sum += p[3] + p[4] + p[5]
        count += 3
        # Sanity-check the stored quaternion components.
        qi = result["quat"][i]
        assert len(qi) == 4
        assert math.isclose(qi[0], 1.0 / nq)
        assert math.isclose(qi[1], p[0] / nq)
        assert math.isclose(qi[2], p[1] / nq)
        assert math.isclose(qi[3], p[2] / nq)
        # Each frame is [R, t] with R a 3x3 rotation and t a 3-vector.
        R, t = result["frames"][i]
        assert len(R) == 3 and all(len(row) == 3 for row in R)
        assert len(t) == 3

    assert math.isclose(result["estimate"], est_sum / count)


def test_alfbk2_edge():
    """Test edge cases."""
    rng = np.random.default_rng(42)
    n, cs = 3, 4
    s = [rng.normal(0, 1, cs).tolist() for _ in range(n)]
    w = [rng.normal(0, 1, cs).tolist() for _ in range(6)]

    # With zero input the leading quaternion component is fixed to 1, so the
    # resulting rotation must be the identity.
    result = alphafold_backbone([[0.0] * cs for _ in range(n)], w)
    assert isinstance(result, dict)
    assert result["n"] == n
    for qi in result["quat"]:
        assert math.isclose(qi[0], 1.0)
        assert math.isclose(qi[1], 0.0)
        assert math.isclose(qi[2], 0.0)
        assert math.isclose(qi[3], 0.0)
    for R, _t in result["frames"]:
        for r in range(3):
            for c in range(3):
                expected = 1.0 if r == c else 0.0
                assert math.isclose(R[r][c], expected)
