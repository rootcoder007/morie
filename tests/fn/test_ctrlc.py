"""ctrlc: many-to-one nonparametric comparison against a control.

Gibbons & Chakraborti 5e, Ch 10 (Tests of the Equality of k Independent
Samples), section 10.7 "Comparisons with a control".
"""

import numpy as np
import pytest

from morie.fn.ctrlc import control_comparison as cmpc


def _groups(shifts, n=25, seed=1501):
    rng = np.random.default_rng(seed)
    return [rng.normal(s, 1.0, n) for s in shifts]


def test_ctrlc_makes_k_minus_one_comparisons_not_k():
    """The control is compared AGAINST, not compared with itself.

    With 4 groups there are 3 comparisons, and `k` reports that count -- not
    the number of groups.
    """
    r = cmpc(_groups([0.0, 0.0, 0.0, 0.0]))
    assert r["k"] == 3
    assert np.asarray(r["p_value"]).size == 3
    assert np.asarray(r["statistic"]).size == 3


def test_ctrlc_identical_groups_are_not_significant():
    r = cmpc(_groups([0.0, 0.0, 0.0]))
    assert np.all(np.asarray(r["p_value"]) > 0.05)


def test_ctrlc_a_clearly_shifted_treatment_is_flagged():
    """Control at 0, treatments at 0 and +3 SD: only the second should fire."""
    r = cmpc(_groups([0.0, 0.0, 3.0], n=30))
    p = np.asarray(r["p_adjusted"])
    assert p[0] > 0.05
    assert p[1] < 0.01


def test_ctrlc_adjustment_never_makes_a_p_value_smaller():
    """Multiplicity control can only be conservative -- that is its job.

    With k-1 comparisons against one control the family-wise error rate is
    inflated, so an unadjusted p is the optimistic one.
    """
    r = cmpc(_groups([0.0, 0.5, 1.0, 1.5, 2.0], n=20))
    raw = np.asarray(r["p_value"])
    adj = np.asarray(r["p_adjusted"])
    assert np.all(adj >= raw - 1e-12)
    assert np.all(adj <= 1.0 + 1e-12)


def test_ctrlc_control_index_selects_which_group_is_the_control():
    """Putting the control last must give the same comparisons as first."""
    g = _groups([0.0, 2.5, 0.0], n=30)
    first = cmpc([g[0], g[1], g[2]], control_index=0)
    moved = cmpc([g[1], g[2], g[0]], control_index=2)
    assert sorted(np.asarray(first["p_value"]).round(10)) == pytest.approx(
        sorted(np.asarray(moved["p_value"]).round(10))
    )


def test_ctrlc_reports_the_control_size_and_the_treatment_sizes():
    """`n` is the per-treatment size array, one entry per comparison."""
    r = cmpc(_groups([0.0, 0.0, 0.0], n=17))
    assert r["control_n"] == 17
    assert np.asarray(r["n"]).tolist() == [17, 17]


def test_ctrlc_statistic_straddles_the_null_value_by_direction():
    """The statistic is a Mann-Whitney U, so it is never negative: direction
    shows up as U above or below its null mean n1*n2/2, not as a sign.

    A treatment shifted up and one shifted down must land on opposite sides
    of that midpoint, which is the property a caller needs in order to read
    the direction of the effect at all.
    """
    n = 30
    null_mean = n * n / 2
    up = float(np.asarray(cmpc(_groups([0.0, 2.0], n=n))["statistic"]).ravel()[0])
    down = float(np.asarray(cmpc(_groups([0.0, -2.0], n=n))["statistic"]).ravel()[0])
    assert up >= 0 and down >= 0, "Mann-Whitney U is non-negative"
    assert (up - null_mean) * (down - null_mean) < 0, "must straddle the null"


def test_ctrlc_default_adjustment_is_bonferroni():
    r = cmpc(_groups([0.0, 0.0, 0.0]))
    assert r["adjust"] == "bonferroni"
    raw = np.asarray(r["p_value"])
    adj = np.asarray(r["p_adjusted"])
    m = raw.size
    assert adj == pytest.approx(np.minimum(raw * m, 1.0))
