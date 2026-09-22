"""Tests for bdgmm2.bound_gmm_alt."""

from morie.fn import _array_core as np

from morie.fn.bdgmm2 import bound_gmm_alt


def test_bdgmm2_basic():
    """Test basic functionality against the documented formula."""
    rng_m = np.random.default_rng(7)
    rng_s = np.random.default_rng(11)

    n = 200
    mbar = rng_m.normal(0, 1, 6)
    sigma = rng_s.uniform(0.5, 2.0, 6)  # strictly positive, same length as mbar

    result = bound_gmm_alt(mbar, sigma, n)

    # The function returns a dict-like RichResult with the documented keys.
    assert isinstance(result, dict)
    assert "S" in result
    assert "t" in result
    assert "xi" in result
    assert "retained" in result
    assert "nretained" in result
    assert "kappa" in result
    assert "n" in result
    assert "J" in result

    # Re-compute every numeric expectation from the documented formula.
    import math

    m_list = list(np.asarray(mbar).ravel())
    s_list = list(np.asarray(sigma).ravel())
    J = len(m_list)
    n_f = float(n)
    k = math.sqrt(math.log(n_f))

    t_expected = [math.sqrt(n_f) * m_list[j] / s_list[j] for j in range(J)]
    xi_expected = [t_expected[j] / k for j in range(J)]
    keep_expected = [1 if xi_expected[j] > -1.0 else 0 for j in range(J)]
    S_expected = sum(
        max(t_expected[j], 0.0) ** 2 for j in range(J) if keep_expected[j]
    )

    assert result["n"] == n_f
    assert result["J"] == J
    assert result["kappa"] == k
    assert result["nretained"] == sum(keep_expected)

    assert len(result["t"]) == J
    assert len(result["xi"]) == J
    assert len(result["retained"]) == J

    for j in range(J):
        assert result["t"][j] == t_expected[j]
        assert result["xi"][j] == xi_expected[j]
        assert result["retained"][j] == keep_expected[j]

    assert result["S"] == S_expected

    # retained is a 0/1 indicator list.
    assert all(v in (0, 1) for v in result["retained"])


def test_bdgmm2_edge():
    """Test edge cases against the documented behaviour."""
    import math

    # A small J=2 case with one retained and one dropped moment.
    mbar = [0.05, -0.4]
    sigma = [1.0, 1.0]
    n = 100

    result = bound_gmm_alt(mbar, sigma, n)

    assert isinstance(result, dict)
    assert result["J"] == 2
    assert result["n"] == float(n)
    assert result["kappa"] == math.sqrt(math.log(float(n)))

    # Independent re-computation for this edge case.
    t = [math.sqrt(float(n)) * mbar[j] / sigma[j] for j in range(2)]
    xi = [t[j] / result["kappa"] for j in range(2)]
    keep = [1 if xi[j] > -1.0 else 0 for j in range(2)]
    S = sum(max(t[j], 0.0) ** 2 for j in range(2) if keep[j])

    assert result["t"] == t
    assert result["xi"] == xi
    assert result["retained"] == keep
    assert result["nretained"] == sum(keep)
    assert result["S"] == S
