"""Gibbons cluster B: runs tests (Ch 3). Truth oracle: brute-force
enumeration of all arrangements, plus the PDF-verified Theorem 3.2.1
constants."""

from itertools import permutations
from math import comb

import numpy as np
import pytest

from morie.fn.gb321 import gibbons_runs_joint_dist
from morie.fn.gb321c import gibbons_marginal_r1
from morie.fn.gb321l import gibbons_distributing_objects
from morie.fn.gb322 import gibbons_total_runs_dist
from morie.fn.gb32l2 import gibbons_vandermonde_id1
from morie.fn.gb32l3 import gibbons_vandermonde_id2
from morie.fn.gb32lu import gibbons_runs_up_down_recur
from morie.fn.gb32mn import gibbons_runs_mean
from morie.fn.gb32vr import gibbons_runs_var
from morie.fn.gb331 import gibbons_run_lengths_dist
from morie.fn.gb332 import gibbons_type1_run_lengths
from morie.fn.gb34mn import gibbons_runs_ud_mean


def _all_arrangements(n1, n2):
    """Every distinct 0/1 sequence with n1 zeros and n2 ones."""
    from itertools import combinations

    n = n1 + n2
    for pos in combinations(range(n), n1):
        seq = np.ones(n, dtype=int)
        seq[list(pos)] = 0
        yield seq


def _run_stats(seq):
    change = np.flatnonzero(np.diff(seq) != 0)
    bounds = np.r_[-1, change, len(seq) - 1]
    lengths = np.diff(bounds)
    types = seq[np.r_[0, change + 1]]
    r1 = int(np.sum(types == 0))
    r2 = int(np.sum(types == 1))
    return r1, r2, lengths, types


def test_joint_and_marginal_and_total_match_enumeration():
    for n1, n2 in ((4, 5), (3, 6), (5, 5)):
        total = comb(n1 + n2, n1)
        joint_emp = {}
        r1_emp = {}
        r_emp = {}
        for seq in _all_arrangements(n1, n2):
            r1, r2, _, _ = _run_stats(seq)
            joint_emp[(r1, r2)] = joint_emp.get((r1, r2), 0) + 1
            r1_emp[r1] = r1_emp.get(r1, 0) + 1
            r_emp[r1 + r2] = r_emp.get(r1 + r2, 0) + 1
        # joint pmf, Theorem 3.2.1 -- every feasible cell exactly
        for (r1, r2), cnt in joint_emp.items():
            assert gibbons_runs_joint_dist(r1, r2, n1, n2)["pmf"] == pytest.approx(
                cnt / total, abs=1e-14
            )
        # infeasible cells are exactly zero
        assert gibbons_runs_joint_dist(1, 3, n1, n2)["pmf"] == 0.0
        # marginal, Corollary 3.2.1
        for r1, cnt in r1_emp.items():
            assert gibbons_marginal_r1(r1, n1, n2)["pmf"] == pytest.approx(
                cnt / total, abs=1e-14
            )
        # total runs, Theorem 3.2.2
        for r, cnt in r_emp.items():
            assert gibbons_total_runs_dist(r, n1, n2)["pmf"] == pytest.approx(
                cnt / total, abs=1e-14
            )
        # moments, eq. 3.2.6 and 3.2.8
        rs = np.array(sorted(r_emp))
        p = np.array([r_emp[r] for r in rs]) / total
        assert gibbons_runs_mean(n1, n2)["mean"] == pytest.approx(np.sum(rs * p))
        assert gibbons_runs_var(n1, n2)["var"] == pytest.approx(
            np.sum(rs**2 * p) - np.sum(rs * p) ** 2, abs=1e-12
        )


def test_run_lengths_distributions_match_enumeration():
    n1, n2 = 4, 4
    total = comb(n1 + n2, n1)
    joint_emp = {}
    type1_emp = {}
    for seq in _all_arrangements(n1, n2):
        r1, r2, lengths, types = _run_stats(seq)
        L1 = tuple(sorted(lengths[types == 0]))
        L2 = tuple(sorted(lengths[types == 1]))
        joint_emp[(L1, L2)] = joint_emp.get((L1, L2), 0) + 1
        type1_emp[L1] = type1_emp.get(L1, 0) + 1
    for (L1, L2), cnt in joint_emp.items():
        assert gibbons_run_lengths_dist(L1, L2)["pmf"] == pytest.approx(
            cnt / total, abs=1e-14
        ), (L1, L2)
    for L1, cnt in type1_emp.items():
        assert gibbons_type1_run_lengths(L1, n2=n2)["pmf"] == pytest.approx(
            cnt / total, abs=1e-14
        ), L1
    with pytest.raises(ValueError):
        gibbons_run_lengths_dist([2, 2], [1, 1, 1, 1, 1])  # |r1-r2| > 1
    with pytest.raises(ValueError):
        gibbons_type1_run_lengths([2, 2], n2=None)


def test_lemmas_hold_and_lemma1_counts_compositions():
    for m, n in ((3, 5), (6, 6), (0, 4), (7, 2)):
        assert gibbons_vandermonde_id1(m, n)["holds"] is True
    for m, n in ((3, 5), (6, 6), (0, 4), (7, 2)):
        assert gibbons_vandermonde_id2(m, n)["holds"] is True
    # Lemma 3.2.1 against direct composition enumeration
    def compositions(n, r):
        if r == 1:
            return 1
        return sum(compositions(n - k, r - 1) for k in range(1, n - r + 2))

    for n, r in ((6, 3), (5, 2), (7, 4)):
        assert gibbons_distributing_objects(n, r)["count"] == compositions(n, r)
    with pytest.raises(ValueError):
        gibbons_distributing_objects(3, 5)


def test_runs_up_down_exact_pmf_matches_the_moment_formulas():
    for n in (4, 5, 6, 7):
        out = gibbons_runs_up_down_recur(n=n)
        assert np.sum(out["pmf"]) == pytest.approx(1.0)
        ref = gibbons_runs_ud_mean(n)
        assert out["mean"] == pytest.approx(ref["mean"], abs=1e-12)  # (2n-1)/3
        assert out["var"] == pytest.approx(ref["var"], abs=1e-12)  # (16n-29)/90
        assert out["support"].max() == n - 1
    # scoring a monotone sequence: R_ud = 1, the smallest possible
    obs = gibbons_runs_up_down_recur(x=[1.0, 2.0, 3.0, 4.0, 5.0])
    assert obs["observed"] == 1
    assert obs["p_le"] == pytest.approx(obs["pmf"][0])
    with pytest.raises(ValueError):
        gibbons_runs_up_down_recur(x=[1.0, 1.0, 2.0])  # ties
    with pytest.raises(ValueError):
        gibbons_runs_up_down_recur(n=15)
