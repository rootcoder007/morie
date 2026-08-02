"""mdrnk: midranks with tie correction (Gibbons & Chakraborti 5e, Ch 5.6.2)."""

from morie.fn import _array_core as np
import pytest

from morie.fn.mdrnk import midranks as mr


def test_mdrnk_untied_sample_is_the_plain_rank():
    assert np.asarray(mr([10.0, 20.0, 30.0])["midranks"]) == pytest.approx([1, 2, 3])


def test_mdrnk_ties_share_the_average_of_their_ranks():
    """(1, 2, 2, 3): the two 2s occupy ranks 2 and 3, so both get 2.5."""
    got = np.asarray(mr([1.0, 2.0, 2.0, 3.0])["midranks"])
    assert got == pytest.approx([1.0, 2.5, 2.5, 4.0])


def test_mdrnk_a_full_tie_gives_everyone_the_same_midrank():
    """n identical values all rank (n+1)/2."""
    n = 5
    assert np.asarray(mr(np.ones(n))["midranks"]) == pytest.approx(np.full(n, 3.0))


def test_mdrnk_midranks_always_sum_to_n_times_n_plus_one_over_two():
    """Averaging tied ranks preserves the total -- that is the point of
    midranks rather than arbitrary tie-breaking."""
    rng = np.random.default_rng(1201)
    for _ in range(20):
        x = rng.integers(0, 5, size=12).astype(float)
        n = x.size
        assert float(np.sum(mr(x)["midranks"])) == pytest.approx(n * (n + 1) / 2)


def test_mdrnk_reports_the_tie_groups():
    r = mr([1.0, 2.0, 2.0, 2.0, 3.0, 3.0])
    ties = dict(r["ties"])
    assert ties[2.0] == 3
    assert ties[3.0] == 2
    assert 1.0 not in ties


def test_mdrnk_tie_correction_is_zero_without_ties_and_positive_with_them():
    """The correction term exists to adjust rank-test variances; it must
    vanish exactly when every observation is distinct."""
    assert mr([3.0, 1.0, 2.0])["tie_correction"] == pytest.approx(0.0)
    assert mr([1.0, 1.0, 2.0])["tie_correction"] > 0.0


def test_mdrnk_is_order_equivariant():
    """Permuting the input permutes the midranks the same way."""
    x = np.array([5.0, 1.0, 3.0, 3.0, 9.0])
    perm = np.array([3, 0, 4, 1, 2])
    a = np.asarray(mr(x)["midranks"])[perm]
    b = np.asarray(mr(x[perm])["midranks"])
    assert a == pytest.approx(b)
