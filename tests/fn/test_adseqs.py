"""Tests for adseqs.admixture_seq."""

import math

from morie.fn import _array_core as np

from morie.fn.adseqs import admixture_seq


def _loglik_formula(G, Q, P):
    """Independent implementation of the log-likelihood from the docstring."""
    I = len(G)
    J = len(G[0]) if I else 0
    K = len(Q[0]) if I else 0
    tot = 0.0
    for i in range(I):
        for j in range(J):
            a = sum(Q[i][k] * P[k][j] for k in range(K))
            b = sum(Q[i][k] * (1.0 - P[k][j]) for k in range(K))
            g = G[i][j]
            if g > 0.0:
                tot += g * math.log(a)
            if 2.0 - g > 0.0:
                tot += (2.0 - g) * math.log(b)
    return tot


def test_adseqs_basic():
    """Test basic functionality with a valid (I, J) genotype matrix."""
    rng = np.random.default_rng(42)
    # Genotypes must be 2-D with values in {0, 1, 2}; generate integer counts.
    G = rng.integers(0, 3, size=(20, 50))
    K = 3
    steps = 25
    result = admixture_seq(G, K=K, steps=steps)
    assert isinstance(result, dict)
    # The function returns these documented keys:
    assert "Q" in result
    assert "P" in result
    assert "loglik" in result
    assert "loglik0" in result
    assert "I" in result
    assert "J" in result
    assert "K" in result
    assert "steps" in result
    assert "method" in result
    # Shape checks for the ancestry proportions and allele frequencies.
    assert len(result["Q"]) == 20
    assert len(result["Q"][0]) == 3
    assert len(result["P"]) == 3
    assert len(result["P"][0]) == 50
    # Reported dimensions and iteration count match the call.
    assert result["I"] == 20
    assert result["J"] == 50
    assert result["K"] == 3
    assert result["steps"] == steps
    # Each row of Q must sum to 1 (mixture proportions for an individual).
    for row in result["Q"]:
        s = sum(row)
        assert abs(s - 1.0) < 1e-9
    # Allele frequencies must lie in (0, 1).
    for k in range(K):
        for j in range(50):
            assert 0.0 < result["P"][k][j] < 1.0
    # Verify the returned log-likelihood against the formula written out
    # with plain arithmetic on the same inputs (independent of the function).
    expected_ll = _loglik_formula(
        [[int(v) for v in row] for row in G],
        result["Q"],
        result["P"],
    )
    assert abs(result["loglik"] - expected_ll) < 1e-9
    # EM should not decrease the log-likelihood for this problem.
    assert result["loglik"] >= result["loglik0"] - 1e-9


def test_adseqs_edge():
    """Test edge cases: deterministic defaults and consistency of result."""
    rng = np.random.default_rng(42)
    G = rng.integers(0, 3, size=(5, 10))
    K = 2
    steps = 10
    result = admixture_seq(G, K=K, steps=steps)
    assert isinstance(result, dict)
    # Deterministic starts must yield a finite initial log-likelihood.
    assert math.isfinite(result["loglik0"])
    # After any number of EM steps the likelihood must remain finite.
    assert math.isfinite(result["loglik"])
    # loglik0 must be reproducible across calls with the same inputs.
    result2 = admixture_seq(G, K=K, steps=steps)
    assert abs(result["loglik0"] - result2["loglik0"]) < 1e-12
    assert abs(result["loglik"] - result2["loglik"]) < 1e-12
