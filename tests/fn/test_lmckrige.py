from morie.fn._rng import random_uniform
from morie.fn.krgsys import krige
from morie.fn.lmckrige import lmc_cokriging, lmc_covariance


def _data(dim=4, n=20, seed=6):
    u = [float(v) for v in random_uniform(n * (dim + 1), seed=seed)]
    P = [tuple(4 * u[(dim + 1) * i + t] for t in range(dim)) for i in range(n)]
    z = [sum(p) + u[(dim + 1) * i + dim] for i, p in enumerate(P)]
    return P, z


def test_one_variable_is_ordinary_kriging_4d():
    P, z = _data()
    lmc = [{"model": "Nug", "B": [[0.2]]}, {"model": "Exp", "range": 2.0, "B": [[1.3]]}]
    Q = [(1, 1, 1, 1), (2.5, 0.5, 3, 1.2)]
    r = lmc_cokriging(z, P, [0] * 20, Q, lmc)
    ref = krige(z, P, Q, [{"model": "Nug", "psill": 0.2}, {"model": "Exp", "psill": 1.3, "range": 2.0}])
    for a, b in zip(r.prediction + r.variance, ref.prediction + ref.variance):
        assert abs(a - b) < 1e-10


def test_intrinsic_model_is_autokrigeable():
    P, z = _data(dim=3, n=12)
    B = [[1.0, 0.8], [0.8, 2.0]]
    lmc = [{"model": "Exp", "range": 1.5, "B": B}]
    z2 = [0.5 * v + 1 for v in z]
    Q = [(1.0, 2.0, 0.5)]
    co = lmc_cokriging(z + z2, P + P, [0] * 12 + [1] * 12, Q, lmc)
    uni = krige(z, P, Q, {"model": "Exp", "psill": 1.0, "range": 1.5})
    assert abs(co.prediction[0] - uni.prediction[0]) < 1e-9 and abs(co.variance[0] - uni.variance[0]) < 1e-9
    assert max(abs(w) for w in co.weights[0][12:]) < 1e-9


def test_weights_and_covariance():
    lmc = [{"model": "Exp", "range": 2.0, "B": [[1.0, 0.6], [0.6, 1.0]]}]
    r = lmc_cokriging([1.0, 2.0, 1.5, 0.5], [(0, 0), (2, 0), (1, 0), (0, 1)], [0, 0, 1, 1], [(1, 1)], lmc)
    w = r.weights[0]
    assert abs(sum(w[:2]) - 1) < 1e-12 and abs(sum(w[2:])) < 1e-12
    assert abs(r.prediction[0] - (w[0] * 1 + w[1] * 2 + w[2] * 1.5 + w[3] * 0.5)) < 1e-12
    import math

    c = lmc_covariance(1.0, lmc)
    assert abs(c[0][1] - 0.6 * math.exp(-0.5)) < 1e-15
