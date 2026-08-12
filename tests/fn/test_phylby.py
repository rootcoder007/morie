"""Tests for phylby (Ronquist & Huelsenbeck 2003, MrBayes 3)."""

import math

from morie.fn.phylby import (bayesian_phylogeny, chain_temperature,
                             clade_credibility, log_posterior,
                             nni_neighbours, phylby, splits_of,
                             swap_acceptance, topology_key)
from morie.fn.phylml import phylml

T = (("a", 0.1, "b", 0.1), 0.1, ("c", 0.1, "d", 0.1), 0.1)
SEQS = {"a": "ACGTACGTAC", "b": "ACGTACGTAC", "c": "TGCATGCATG",
        "d": "TGCATGCATG"}
SIG = {"a": "ACGTACGTACGTACGTACGT", "b": "ACGTACGTACGTACGTACGA",
       "c": "TGCATGCATGCATGCATGCA", "d": "TGCATGCATGCATGCATGCT"}


def test_chain_temperatures():
    assert abs(chain_temperature(0) - 1.0) < 1e-15
    assert abs(chain_temperature(3, 0.2) - 1.0 / 1.6) < 1e-15
    assert all(chain_temperature(j, 0.2) > chain_temperature(j + 1, 0.2)
               for j in range(5))


def test_the_swap_acceptance_rule():
    assert abs(swap_acceptance(1.0, 1.0, -100.0, -50.0) - 1.0) < 1e-15
    assert abs(swap_acceptance(1.0, 0.5, -100.0, -50.0) - 1.0) < 1e-15
    got = swap_acceptance(1.0, 0.5, -50.0, -100.0)
    assert abs(got - math.exp(0.5 * -50.0)) < 1e-12
    assert 0.0 <= got <= 1.0


def test_splits_and_topology_keys():
    sp = splits_of(T)
    assert len(sp) == 1
    assert sorted(list(sp)[0]) == ["a", "b"]
    assert topology_key(T) == topology_key(
        (("a", 9.9, "b", 0.5), 2.0, ("c", 0.3, "d", 7.7), 1.0))


def test_nni_reaches_every_four_taxon_topology():
    nb = nni_neighbours(T)
    keys = set(topology_key(t) for t in nb) | {topology_key(T)}
    assert len(keys) == 3
    assert all(topology_key(t) != topology_key(T) for t in nb)


def test_the_posterior_is_likelihood_plus_priors():
    post = log_posterior(T, SEQS, branch_prior_mean=0.1)
    assert abs(post["loglik"] - phylml(T, SEQS)["log_likelihood"]) < 1e-12
    lam = 1.0 / 0.1
    want = sum(math.log(lam) - lam * 0.1 for _ in range(6))
    assert abs(post["logprior"] - want) < 1e-12
    assert abs(post["logpost"] - (post["loglik"] +
                                  post["logprior"])) < 1e-12
    hot = log_posterior(T, SEQS, branch_prior_mean=0.1, temperature=0.5)
    assert abs(hot["logpost"] - 0.5 * post["logpost"]) < 1e-12


def test_uninformative_data_gives_the_uniform_topology_prior():
    flat = dict((t, "AAAAAAAAAAAAAAAA") for t in "abcd")
    res = phylby(flat, n_iter=3000, n_chains=2, sample_every=5, n_runs=2,
                 seed=11)
    probs = {}
    for t in res["samples"]:
        k = topology_key(t)
        probs[k] = probs.get(k, 0) + 1
    n = float(len(res["samples"]))
    share = sorted(v / n for v in probs.values())
    assert len(probs) == 3
    assert max(share) < 0.55 and min(share) > 0.15


def test_a_planted_topology_is_recovered():
    res = phylby(SIG, n_iter=3000, n_chains=3, sample_every=5, n_runs=2,
                 seed=5)
    assert res["map_topology"] == topology_key(T)
    assert res["map_probability"] > 0.9
    assert res["clade_credibility"][("a", "b")] > 0.95
    assert res["asdsf"] < 0.05
    assert 0.0 < res["swap_rate"] <= 1.0
    assert 0.0 < res["acceptance"] < 1.0
    assert abs(res["temperatures"][0] - 1.0) < 1e-15


def test_partitions_carry_their_own_rate():
    parts = ["fast"] * 10 + ["slow"] * 10
    same = log_posterior(T, SIG, partitions=parts,
                         rates={"fast": 1.0, "slow": 1.0})
    diff = log_posterior(T, SIG, partitions=parts,
                         rates={"fast": 3.0, "slow": 0.3})
    assert abs(same["loglik"] - diff["loglik"]) > 1e-6
    assert abs(same["loglik"] -
               phylml(T, SIG)["log_likelihood"]) < 1e-9
    run = phylby(SIG, n_iter=400, n_chains=2, partitions=parts, n_runs=1,
                 seed=2)
    assert run["n_samples"] > 0


def test_clade_credibility():
    cred = clade_credibility([T, T])
    assert abs(cred[("a", "b")] - 1.0) < 1e-15


def test_validation():
    for call in (lambda: phylby({"a": "AC", "b": "AC"}),
                 lambda: phylby({"a": "AC", "b": "ACG", "c": "AC",
                                 "d": "AC"}),
                 lambda: phylby(dict((t, "") for t in "abcd")),
                 lambda: phylby(SIG, n_iter=0),
                 lambda: phylby(SIG, n_chains=0),
                 lambda: phylby(SIG, n_runs=0),
                 lambda: phylby(SIG, swap_every=0),
                 lambda: phylby(SIG, sample_every=0),
                 lambda: phylby(SIG, burnin=10 ** 9),
                 lambda: phylby(SIG, partitions=["a"]),
                 lambda: chain_temperature(-1),
                 lambda: chain_temperature(1, lam=-0.5),
                 lambda: log_posterior(T, SEQS, branch_prior_mean=0.0),
                 lambda: log_posterior(T, SEQS, rate=0.0),
                 lambda: log_posterior((("a", -1.0, "b", 0.1), 0.1,
                                        ("c", 0.1, "d", 0.1), 0.1),
                                       SEQS),
                 lambda: log_posterior((("a", 0.1, "b", 0.1), 0.1,
                                        ("c", 0.1, "d", 0.1)), SEQS),
                 lambda: log_posterior(T, SEQS, partitions=["x"] * 10,
                                       rates={"x": 0.0}),
                 lambda: clade_credibility([])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert bayesian_phylogeny is phylby
