"""Tests for rappor.

The generated tests these replace passed 100-long normal draws as the
privacy parameters f and q, which _check_params rejects immediately, so
they never reached the algorithm. Every assertion here is against a
closed form printed in Erlingsson, Pihur & Korolova (2014) or against a
property of the estimator that a wrong implementation would break.
"""

import math

import pytest

from morie.fn.rappor import (rappor, rappor_decode, rappor_encode,
                             rappor_epsilon, rappor_star_probs)


def test_epsilon_infinity_matches_theorem_1():
    r"""eps_inf = 2h ln((1 - f/2) / (f/2)).

    At f = 1/2 the ratio is 3, so eps_inf = 2h ln 3 exactly. Two h
    values pin the linearity as well as the constant.
    """
    assert rappor_epsilon(1, 0.5)["eps_infinity"] == pytest.approx(
        2.0 * math.log(3.0))
    assert rappor_epsilon(2, 0.5)["eps_infinity"] == pytest.approx(
        4.0 * math.log(3.0))
    # f -> 1 makes the permanent report pure noise, so eps_inf -> 0.
    assert rappor_epsilon(2, 1.0)["eps_infinity"] == pytest.approx(0.0)
    # Smaller f is weaker longitudinal privacy, i.e. larger epsilon.
    eps = [rappor_epsilon(2, f)["eps_infinity"] for f in (0.9, 0.5, 0.1)]
    assert eps[0] < eps[1] < eps[2]


def test_star_probs_lemma_1():
    """q* = f(p+q)/2 + (1-f)q, p* = f(p+q)/2 + (1-f)p."""
    qs, ps = rappor_star_probs(0.5, 0.5, 0.75)
    assert qs == pytest.approx(0.25 * 1.25 + 0.5 * 0.75)
    assert ps == pytest.approx(0.25 * 1.25 + 0.5 * 0.5)
    # f = 0 removes the PRR entirely: the effective probabilities are
    # just q and p.
    qs0, ps0 = rappor_star_probs(0.0, 0.3, 0.8)
    assert (qs0, ps0) == pytest.approx((0.8, 0.3))
    # f = 1 collapses them onto each other, so no signal survives.
    qs1, ps1 = rappor_star_probs(1.0, 0.3, 0.8)
    assert qs1 == pytest.approx(ps1)


def test_epsilon_one_is_zero_when_no_signal_survives():
    """f = 1 gives q* = p*, so Theorem 2's eps_1 is 0."""
    out = rappor_epsilon(4, 1.0, p=0.3, q=0.8)
    assert out["eps_1"] == pytest.approx(0.0)
    # and with the PRR off, eps_1 is h log(q(1-p)/(p(1-q))).
    out0 = rappor_epsilon(3, 1e-12, p=0.25, q=0.75)
    assert out0["eps_1"] == pytest.approx(
        3.0 * math.log((0.75 * 0.75) / (0.25 * 0.25)), rel=1e-6)


def test_decode_is_unbiased_for_the_basic_variant():
    """The Sec. 4 estimator recovers the true one-hot counts.

    "basic" is one-hot over the alphabet, so bit i counts exactly the
    clients holding symbol i and the decoded t is directly comparable
    with the population. 20000 clients over 3 symbols with f = 0.2 keeps
    the per-bit standard error near 1% of N.
    """
    vals = (["a"] * 10000) + (["b"] * 6000) + (["c"] * 4000)
    enc = rappor_encode(vals, f=0.2, p=0.25, q=0.75,
                        variant="basic", seed=7)
    assert enc["alphabet"] == ["a", "b", "c"]
    dec = rappor_decode(enc["counts"], enc["cohort_sizes"],
                        f=0.2, p=0.25, q=0.75)
    est = dec["estimate"][0]
    for got, want in zip(est, (10000.0, 6000.0, 4000.0)):
        assert abs(got - want) < 0.03 * 20000


def test_decode_shift_and_denominator_are_the_printed_ones():
    dec = rappor_decode([[0]], [0], f=0.5, p=0.25, q=0.75)
    assert dec["denominator"] == pytest.approx(0.5 * 0.5)
    assert dec["shift"] == pytest.approx(0.25 + 0.5 * 0.75 / 2
                                         - 0.5 * 0.25 / 2)


def test_decode_refuses_a_signal_free_configuration():
    """(1 - f)(q - p) = 0 leaves nothing to invert."""
    with pytest.raises(ValueError):
        rappor_decode([[1, 2]], [3], f=1.0, p=0.25, q=0.75)
    with pytest.raises(ValueError):
        rappor_decode([[1, 2]], [3], f=0.5, p=0.5, q=0.5)


def test_permanent_response_is_memoized_per_client():
    """One client reporting the same value repeatedly must reuse B'.

    That memoization is the whole longitudinal defence: without it an
    attacker averaging the reports recovers B exactly. The one-time
    variant reports B' itself, so memoization is directly visible as
    identical reports; a fresh PRR per report would not be identical at
    f = 0.5 over 16 bits.
    """
    enc = rappor_encode(["x"] * 6, k=16, h=2, f=0.5, variant="one-time",
                        seed=3, client_ids=["c1"] * 6)
    reports = enc["reports"]
    assert all(r == reports[0] for r in reports)

    # Different clients get independent permanent responses, so at least
    # one of six differs from the first.
    many = rappor_encode(["x"] * 6, k=16, h=2, f=0.5, variant="one-time",
                         seed=3, client_ids=["c%d" % i for i in range(6)])
    assert any(r != many["reports"][0] for r in many["reports"])

    # A client changing its value gets a new B' -- the memo is keyed on
    # the pair, not on the client alone.
    two = rappor_encode(["x", "y"], k=16, h=2, f=0.5, variant="one-time",
                        seed=3, client_ids=["c1", "c1"])
    assert two["reports"][0] != two["reports"][1]


def test_full_variant_rerandomizes_the_instantaneous_report():
    """Step 3 is fresh every time even when B' is memoized -- that is
    what stops a single report from identifying the client."""
    enc = rappor_encode(["x"] * 20, k=16, h=2, f=0.5, p=0.5, q=0.75,
                        variant="full", seed=3, client_ids=["c1"] * 20)
    assert any(r != enc["reports"][0] for r in enc["reports"])


def test_client_keeps_one_cohort():
    """Sec. 3.1: a client is assigned a cohort and keeps it."""
    enc = rappor_encode(["x"] * 30, k=8, h=2, cohorts=4, seed=5,
                        client_ids=["c1"] * 15 + ["c2"] * 15)
    per = {}
    for cid, j in zip(enc["client_ids"], enc["cohort_of"]):
        per.setdefault(cid, set()).add(j)
    assert all(len(v) == 1 for v in per.values())


def test_client_ids_length_is_checked():
    with pytest.raises(ValueError):
        rappor_encode(["a", "b"], client_ids=["only-one"])


def test_bloom_positions_are_deterministic_and_cohort_specific():
    a = rappor_encode(["v"], k=32, h=2, f=1e-12, p=1e-12, q=1.0 - 1e-12,
                      cohorts=1, seed=1)
    b = rappor_encode(["v"], k=32, h=2, f=1e-12, p=1e-12, q=1.0 - 1e-12,
                      cohorts=1, seed=2)
    # Same value, same cohort, negligible noise: the same bits are set.
    assert a["counts"] == b["counts"]
    assert sum(a["counts"][0]) <= 2


def test_rejects_out_of_range_parameters():
    for bad in ({"f": -0.1}, {"f": 2.5}, {"p": 1.5}, {"q": -0.2}):
        with pytest.raises(ValueError):
            rappor_encode(["a"], **bad)
    with pytest.raises(ValueError):
        rappor_encode([])
    with pytest.raises(ValueError):
        rappor_encode(["a"], variant="nope")
    with pytest.raises(ValueError):
        rappor_epsilon(0, 0.5)


def test_rappor_alias_is_the_encoder():
    assert rappor is rappor_encode
