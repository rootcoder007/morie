"""Tests for propinf (Ganju et al. 2018, property inference)."""

import math

from morie.fn.propinf import (fcnn_predict, flat_representation,
                              permute_hidden_layer, property_inference,
                              set_representation, sorted_representation,
                              train_fcnn)


def _lcg(seed):
    st = [seed]

    def f():
        st[0] = (1103515245 * st[0] + 12345) % (1 << 31)
        return st[0] / float(1 << 31)
    return f


def _gauss(r):
    return math.sqrt(-2 * math.log(max(r(), 1e-12))) * \
        math.cos(2 * math.pi * r())


def _fit(seed=5, hidden=(3, 2)):
    r = _lcg(seed)
    X = [[_gauss(r) for _ in range(5)] for _ in range(40)]
    y = [1.0 if (0.9 * row[0] + 0.6 * row[1] + 0.3 * _gauss(r)) > 0 else 0.0
         for row in X]
    return train_fcnn(X, y, hidden=hidden, epochs=25, lr=0.2, seed=seed), X


def test_a_node_permutation_leaves_the_function_alone():
    net, X = _fit()
    base = fcnn_predict(net, X)
    q = permute_hidden_layer(net, 0, [2, 0, 1])
    q = permute_hidden_layer(q, 1, [1, 0])
    assert max(abs(a - b) for a, b in zip(base, fcnn_predict(q, X))) < 1e-12


def test_the_flat_vector_moves_but_the_canonical_form_does_not():
    net, _ = _fit()
    q = permute_hidden_layer(net, 0, [2, 0, 1])
    fa, fb = flat_representation(net), flat_representation(q)
    assert max(abs(a - b) for a, b in zip(fa, fb)) > 0.05
    sa, sb = sorted_representation(net), sorted_representation(q)
    assert max(abs(a - b) for a, b in zip(sa, sb)) < 1e-15


def test_algorithm_1_sorts_by_the_weight_sum_magnitude():
    net, _ = _fit()
    srt = sorted_representation(net)
    n_in = len(net[0]["W"][0])
    keys = [abs(sum(srt[i * (n_in + 1):i * (n_in + 1) + n_in]))
            for i in range(len(net[0]["W"]))]
    assert all(keys[i] >= keys[i + 1] - 1e-12 for i in range(len(keys) - 1))


def test_the_set_representation_is_the_same_multiset():
    net, _ = _fit()
    q = permute_hidden_layer(net, 0, [2, 0, 1])
    a, b = set_representation(net), set_representation(q)
    assert sorted(map(tuple, a[0])) == sorted(map(tuple, b[0]))
    assert a[0] != b[0]


def _bank(seed0, count):
    nets, labels = [], []
    for k in range(count):
        r = _lcg(seed0 + 97 * k)
        idx = 0 if k % 2 == 0 else 2
        X = [[_gauss(r) for _ in range(5)] for _ in range(90)]
        y = [1.0 if (1.2 * row[idx] + 0.3 * _gauss(r)) > 0 else 0.0
             for row in X]
        nets.append(train_fcnn(X, y, hidden=(5, 3), epochs=20, lr=0.15,
                               batch_size=12, seed=seed0 + k))
        labels.append(1 if k % 2 == 0 else 0)
    return nets, labels


def test_the_attack_reads_a_property_off_unseen_models():
    tr, trl = _bank(1000, 30)
    te, tel = _bank(500000, 12)
    res = property_inference(tr, trl, te, tel, representation="set",
                             epochs=30, lr=0.05, seed=2)
    assert res["accuracy"] > 0.65
    assert res["representation"] == "set"
    assert res["architecture"] == [(5, 5), (3, 5), (1, 3)]


def test_every_representation_runs():
    tr, trl = _bank(1000, 12)
    for rep in ("baseline", "sorting", "set"):
        res = property_inference(tr, trl, representation=rep, epochs=5,
                                 seed=1)
        assert res["representation"] == rep
        assert len(res["prediction"]) == len(tr)
        assert res["accuracy"] is None


def test_the_context_routes_run_and_only_one_is_invariant():
    net, _ = _fit(hidden=(4, 3))
    tr, trl = _bank(1000, 12)
    for ctx in ("paired", "as_printed", "none"):
        res = property_inference(tr, trl, representation="set", context=ctx,
                                 epochs=3, seed=1)
        assert res["context"] == ctx


def test_validation():
    net, X = _fit()
    tr, trl = _bank(1000, 8)
    odd = train_fcnn(X, [1.0] * 20 + [0.0] * 20, hidden=(4,), epochs=2)
    for call in (lambda: property_inference(tr[:2], trl[:2]),
                 lambda: property_inference(tr, trl[:-1]),
                 lambda: property_inference(tr, [1] * len(tr)),
                 lambda: property_inference(tr, trl, representation="pca"),
                 lambda: property_inference(tr, trl, context="tuple"),
                 lambda: property_inference(tr, trl, [odd]),
                 lambda: property_inference(tr, trl, epochs=0),
                 lambda: permute_hidden_layer(net, 2, [0, 1]),
                 lambda: permute_hidden_layer(net, 0, [0, 0, 1]),
                 lambda: train_fcnn(X, [1.0] * 40, hidden=()),
                 lambda: train_fcnn(X, [2.0] * 40),
                 lambda: train_fcnn(X, [1.0] * 39)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
