"""Tests for dpsng.dp_singularity_test.

Anchored on Ghosal and van der Vaart (2017) Proposition 4.8, read from
the corpus PDF: the D_i are independent Bernoulli(M/(M+i-1)), so K_n is
Poisson-binomial with

    E(K_n)   = sum_i M/(M+i-1)
    var(K_n) = sum_i M(i-1)/(M+i-1)^2

The convolved exact pmf must reproduce both moments and sum to 1 -- a
check on the convolution that is independent of the closed forms, and
vice versa.
"""

import pytest

from morie.fn.dpsng import dp_singularity_test

PARTITION = [1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7, 8, 8, 9, 10]


def _pmf(M, n):
    p = [M / (M + i - 1.0) for i in range(1, n + 1)]
    pmf = [1.0]
    for pi in p:
        nxt = [0.0] * (len(pmf) + 1)
        for k, v in enumerate(pmf):
            nxt[k] += v * (1 - pi)
            nxt[k + 1] += v * pi
        pmf = nxt
    return pmf


def test_exact_pmf_reproduces_proposition_4_8_moments():
    res = dp_singularity_test(PARTITION, alpha=2.0)
    pmf = _pmf(2.0, 16)
    mean = sum(k * v for k, v in enumerate(pmf))
    var = sum((k - mean) ** 2 * v for k, v in enumerate(pmf))
    assert sum(pmf) == pytest.approx(1.0, abs=1e-12)
    assert res["E_K"] == pytest.approx(mean, abs=1e-12)
    assert res["var_K"] == pytest.approx(var, abs=1e-12)


def test_counts_distinct_labels():
    assert dp_singularity_test(PARTITION, alpha=1.0)["K"] == 10
    assert dp_singularity_test(PARTITION, alpha=1.0)["n"] == 16


def test_huge_concentration_puts_everyone_in_their_own_cluster():
    res = dp_singularity_test(list(range(12)), alpha=1e6)
    assert res["E_K"] == pytest.approx(12.0, abs=1e-3)
    assert res["K"] == 12
    assert res["p_value"] > 0.5


def test_tiny_concentration_expects_one_cluster():
    res = dp_singularity_test([0] * 12, alpha=1e-6)
    assert res["E_K"] == pytest.approx(1.0, abs=1e-4)
    assert res["K"] == 1


def test_p_value_is_a_probability_and_flags_a_surprising_count():
    surprising = dp_singularity_test(list(range(16)), alpha=0.1)
    assert 0.0 <= surprising["p_value"] <= 1.0
    assert surprising["p_value"] < 1e-6  # 16 clusters under alpha=0.1 is absurd


def test_bare_count_form_needs_n():
    assert dp_singularity_test(10, alpha=2.0, n=16)["K"] == 10
    with pytest.raises(ValueError):
        dp_singularity_test(10, alpha=2.0)


def test_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        dp_singularity_test(PARTITION, alpha=0.0)
    with pytest.raises(ValueError):
        dp_singularity_test(20, alpha=1.0, n=16)
