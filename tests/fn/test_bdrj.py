"""Tests for bdrj.backdoor_adjustment_formula (Pearl 2009, Thm 3.3.2)."""

import numpy as np
import pytest

from morie.fn.bdrj import backdoor_adjustment_formula


def _simpson():
    """A confounded table where stratifying reverses the raw comparison.

    Z drives both treatment and outcome. Within each stratum treatment
    helps; pooled, it appears to hurt -- Simpson's paradox. Back-door
    adjustment on Z must recover the within-stratum direction.
    """
    X, Y, Z = [], [], []

    def add(z, x, y, k):
        X.extend([x] * k)
        Y.extend([y] * k)
        Z.extend([z] * k)

    # Stratum 0: mostly untreated, high baseline recovery.
    add(0, 1, 1, 81); add(0, 1, 0, 6)      # treated   93%
    add(0, 0, 1, 234); add(0, 0, 0, 36)    # untreated 87%
    # Stratum 1: mostly treated, low baseline recovery.
    add(1, 1, 1, 192); add(1, 1, 0, 71)    # treated   73%
    add(1, 0, 1, 55); add(1, 0, 0, 25)     # untreated 69%
    return np.array(X), np.array(Y), np.array(Z)


def test_stratum_conditionals_are_reweighted_by_the_population_p_z():
    X, Y, Z = _simpson()
    res = backdoor_adjustment_formula(X, Y, Z)
    n = X.size
    p0 = np.mean(Z == 0)
    p1 = 1 - p0
    # P(Y=1 | do(X=1)) = sum_z P(Y=1 | X=1, Z=z) P(Z=z)
    want = p0 * (81 / 87) + p1 * (192 / 263)
    assert res["distribution"][1][1] == pytest.approx(want, rel=1e-12)
    assert res["n"] == n


def test_adjustment_reverses_the_naive_comparison():
    """The paradox: pooled, treatment looks worse; adjusted, it helps."""
    X, Y, Z = _simpson()
    naive_t = np.mean(Y[X == 1] == 1)
    naive_c = np.mean(Y[X == 0] == 1)
    assert naive_t < naive_c, "pooled comparison favours control"
    d = backdoor_adjustment_formula(X, Y, Z)["distribution"]
    assert d[1][1] > d[0][1], "after adjustment treatment helps"


def test_each_interventional_distribution_sums_to_one():
    X, Y, Z = _simpson()
    for row in backdoor_adjustment_formula(X, Y, Z)["distribution"].values():
        assert sum(row.values()) == pytest.approx(1.0)


def test_p_z_is_the_marginal_and_sums_to_one():
    X, Y, Z = _simpson()
    res = backdoor_adjustment_formula(X, Y, Z)
    assert float(np.sum(res["p_z"])) == pytest.approx(1.0)
    assert len(res["strata"]) == 2


def test_adjustment_equals_the_raw_conditional_when_z_is_independent():
    """With Z unrelated to X, reweighting changes nothing much."""
    rng = np.random.default_rng(3)
    n = 4000
    X = rng.integers(0, 2, n)
    Z = rng.integers(0, 2, n)               # independent of X
    Y = (rng.random(n) < 0.3 + 0.4 * X).astype(int)
    d = backdoor_adjustment_formula(X, Y, Z)["distribution"]
    assert d[1][1] == pytest.approx(np.mean(Y[X == 1] == 1), abs=0.03)


def test_multi_column_z_forms_joint_strata():
    rng = np.random.default_rng(4)
    n = 2000
    Z = rng.integers(0, 2, (n, 2))
    X = rng.integers(0, 2, n)
    Y = rng.integers(0, 2, n)
    res = backdoor_adjustment_formula(X, Y, Z)
    assert len(res["strata"]) == 4


def test_a_single_target_can_be_requested():
    X, Y, Z = _simpson()
    assert list(backdoor_adjustment_formula(X, Y, Z, x=1)["distribution"]) == [1]


def test_empty_stratum_cells_are_reported_not_silently_zero():
    X = np.array([0, 0, 1, 1, 1])
    Y = np.array([0, 1, 1, 0, 1])
    Z = np.array([0, 0, 0, 1, 1])   # no X=0 unit in stratum 1
    res = backdoor_adjustment_formula(X, Y, Z)
    assert (0, "1") in res["incomplete_strata"]


def test_validates_inputs():
    X, Y, Z = _simpson()
    with pytest.raises(ValueError, match="share a length"):
        backdoor_adjustment_formula(X, Y[:-1], Z)
    with pytest.raises(ValueError, match="does not occur in X"):
        backdoor_adjustment_formula(X, Y, Z, x=7)
    with pytest.raises(ValueError, match="must not be empty"):
        backdoor_adjustment_formula(np.array([]), np.array([]), np.array([]))
