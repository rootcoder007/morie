"""FDA by optimal scoring (ESL 12.5.1) against mda::fda(method = polyreg)."""

import math

from morie.fn.eslfda import esl_fda


def test_fda_equals_mda():
    i = list(range(1, 61))
    X = [[math.sin(t) + 0.2 * (t % 3), math.cos(3 * t) + 0.4 * (t % 3), math.sin(2 * t)] for t in i]
    g = [t % 3 for t in i]
    r = esl_fda(X, g, query=[[0.2, 0.5, 0.1], [1.2, 1.0, -0.3]])
    assert all(abs(a - b) < 1e-12 for a, b in zip(r["eigenvalues"], (0.264095178837929, 0.00602149780425885)))
    assert r["prediction"] == [1, 2]
    # mda's variates are ours divided by sqrt(alpha^2 (1 - alpha^2)), up to sign
    s = [math.sqrt(a * (1 - a)) for a in r["eigenvalues"]]
    for got, ref in zip(
        r["variates"], ((0.156612005685254, 0.148021566044603), (1.321466105047347, -0.888557364204975))
    ):
        assert all(abs(abs(v / sc) - abs(w)) < 1e-10 for v, sc, w in zip(got, s, ref))
    train = "0 2 0 2 0 1 1 2 0 2 1 0 2 2 1 0 2 0 2 1 2 0 2 0 2 2 2 0 2 0 1 2 1 1 2 0 0 2 0 2 0 2 0 2 0 2 0 2 0 2 0 2 1 0 0 1 1 1 2 0"
    assert esl_fda(X, g)["prediction"] == [int(v) for v in train.split()]
