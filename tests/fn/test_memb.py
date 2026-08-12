"""Tests for memb (Shokri et al. 2017, membership inference)."""

from morie.fn import _array_core as np

from morie.fn.memb import (knn_trainer, logistic_trainer, memb,
                           membership_inference, precision_recall,
                           synthesize, synthesize_marginals, synthesize_noisy)

D = 10


def _gen(n, seed, noise=False):
    r = np.random.default_rng(seed)
    X = [[1.0 if r.random() < 0.5 else 0.0 for _ in range(D)]
         for _ in range(n)]
    if noise:
        y = [1 if r.random() < 0.5 else 0 for _ in X]
    else:
        y = [1 if sum(x[:3]) >= 2 else 0 for x in X]
    return X, y


def _climbable(rows):
    out = []
    for x in rows:
        p = min(max(sum(x[:D // 2]) / float(D // 2), 1e-6), 1 - 1e-6)
        out.append([1.0 - p, p])
    return out


def test_synthesis_returns_a_confident_record():
    x = synthesize(_climbable, 1, D, conf_min=0.7, iter_max=800, seed=3)
    assert x is not None
    y = _climbable([x])[0]
    assert y[1] > 0.7 and y[1] == max(y)


def test_synthesis_gives_up_when_it_must():
    def flat(rows):
        return [[0.5, 0.5] for _ in rows]
    assert synthesize(flat, 1, D, conf_min=0.9, iter_max=50,
                      seed=1) is None


def test_neighbourhood_size_halves_after_rejections():
    seen = {"first": None}
    dists = []

    def picky(rows):
        x = list(rows[0])
        if seen["first"] is None:
            seen["first"] = x
            return [[0.1, 0.9]]
        dists.append(sum(1 for a, b in zip(x, seen["first"]) if a != b))
        return [[0.9, 0.1]]

    synthesize(picky, 1, D, k_max=8, k_min=1, rej_max=2, conf_min=0.99,
               iter_max=40, seed=13)
    assert set(dists) <= {8, 4, 2, 1}
    assert dists[0] == 8 and 1 in dists


def test_marginal_synthesis_matches_marginals_and_breaks_the_joint():
    X, _ = _gen(200, 5)
    dup = [row + [row[0]] for row in X]
    syn = synthesize_marginals(dup, 800, seed=4)
    for j in (0, 3):
        assert abs(sum(r[j] for r in syn) / len(syn) -
                   sum(r[j] for r in dup) / len(dup)) < 0.08
    agree = sum(1 for r in syn if r[0] == r[-1]) / float(len(syn))
    assert abs(agree - 0.5) < 0.08


def test_noisy_synthesis_flips_the_requested_fraction():
    X, _ = _gen(200, 5)
    noisy = synthesize_noisy(X, fraction=0.2, seed=6)
    flipped = sum(1 for a, b in zip(X, noisy)
                  for u, v in zip(a, b) if u != v) / float(len(X) * D)
    assert abs(flipped - 0.2) < 0.05
    assert synthesize_noisy(X, fraction=0.0, seed=6) == X


def test_precision_recall_by_hand():
    pr = precision_recall([1, 1, 0, 0, 1], [1, 0, 0, 1, 1])
    assert abs(pr["precision"] - 2.0 / 3.0) < 1e-12
    assert abs(pr["recall"] - 2.0 / 3.0) < 1e-12
    assert pr["tp"] == 2 and pr["fp"] == 1 and pr["fn"] == 1


def test_attack_finds_a_memorising_target():
    memoriser = knn_trainer(k=1)
    tX, ty = _gen(60, 21, noise=True)
    target = memoriser(tX, ty)
    oX, oy = _gen(60, 22, noise=True)
    shadows = []
    for k in range(4):
        shadows.append(_gen(60, 300 + k, noise=True) +
                       _gen(60, 400 + k, noise=True))
    res = memb(target, shadows, (tX, ty), (oX, oy), train_fn=memoriser,
               attack_train_fn=logistic_trainer(l2=1e-3, epochs=400))
    assert res["metrics"]["accuracy"] > 0.65
    assert res["attack_train_size"] == 4 * 120
    assert res["attack_classes"] == [0, 1]


def test_attack_is_at_chance_without_overfitting():
    reg = logistic_trainer(l2=0.5, epochs=400, lr=0.5)
    gX, gy = _gen(400, 31)
    flat = reg(gX, gy)
    hX, hy = _gen(200, 32)
    shadows = [_gen(400, 500 + k) + _gen(200, 600 + k) for k in range(4)]
    res = memb(flat, shadows, (gX[:200], gy[:200]), (hX, hy),
               train_fn=reg,
               attack_train_fn=logistic_trainer(l2=1e-3, epochs=400))
    assert abs(res["metrics"]["accuracy"] - 0.5) < 0.1


def test_validation():
    tX, ty = _gen(20, 1)
    target = knn_trainer()(tX, ty)
    for call in (lambda: memb(target, [], (tX, ty), (tX, ty)),
                 lambda: synthesize_noisy(tX, fraction=1.5),
                 lambda: synthesize_marginals([], 5),
                 lambda: synthesize(_climbable, 1, 0),
                 lambda: synthesize(_climbable, 1, D, conf_min=1.0)):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert membership_inference is memb
