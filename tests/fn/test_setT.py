"""Tests for setT (Set Transformer PMA pooling)."""

from morie.fn.setT import setT, set_transformer

Z = [[1.0, 0.0], [0.0, 2.0], [0.5, 0.5], [2.0, 1.0]]
S1 = [[0.3, 0.7]]
PARAMS = {
    "Wq": [[1.0, 0.0], [0.0, 1.0]],
    "Wk": [[0.5, 0.2], [0.1, 0.8]],
    "Wv": [[1.0, 0.3], [0.2, 1.0]],
    "W1": [[0.4, -0.2], [0.3, 0.6]],
    "b1": [0.1, -0.1],
    "W2": [[0.7, 0.1], [-0.3, 0.5]],
    "b2": [0.05, -0.05],
}


def test_sett_permutation_invariance_anchor():
    # Lee et al. (2019): PMA pools the set through K/V only, so any
    # permutation of the rows of Z gives the IDENTICAL output.
    a = setT(Z, S1, PARAMS)
    perm = [Z[2], Z[0], Z[3], Z[1]]
    b = setT(perm, S1, PARAMS)
    assert a["output"] == b["output"]
    assert a["k"] == 1 and len(a["output"]) == 1


def test_sett_shapes_and_attention():
    S2 = [[0.3, 0.7], [-0.5, 0.1]]
    r = setT(Z, S2, PARAMS)
    assert r["k"] == 2 and len(r["output"]) == 2
    # pooling attention: k rows over n set elements, each a softmax
    assert len(r["attention"]) == 2 and len(r["attention"][0]) == 4
    for row in r["attention"]:
        assert abs(sum(row) - 1.0) < 1e-12
    try:
        set_transformer(X=Z, k=1)
        raise AssertionError("set_transformer without S/params must raise")
    except ValueError:
        pass
    a = set_transformer(X=Z, S=S1, params=PARAMS)
    b = setT(Z, S1, PARAMS)
    assert a["output"] == b["output"]
