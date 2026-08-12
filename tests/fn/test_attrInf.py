"""Tests for attrInf (Fredrikson, Jha & Ristenpart 2015)."""

from morie.fn.attrInf import (attrInf, attribute_inference,
                              confusion_error, map_invert, tree_paths,
                              tree_predict, wbwc_invert)

TREE = {"feature": 0, "branches": {
    1: {"label": 0, "count": 10.0},
    0: {"feature": 1, "branches": {
        1: {"label": 1, "count": 80.0},
        0: {"label": 0, "count": 10.0}}}}}
C = [[90.0, 10.0], [20.0, 80.0]]
PRIORS = {0: {0: 0.5, 1: 0.5}, 1: {0: 0.3, 1: 0.7}}
FLAT = {0: {0: 0.5, 1: 0.5}, 1: {0: 0.5, 1: 0.5}}


def test_the_tree_and_its_paths():
    assert [tree_predict(TREE, [a, b])
            for a in (0, 1) for b in (0, 1)] == [0, 1, 0, 0]
    paths = tree_paths(TREE)
    assert len(paths) == 3
    assert sorted(p["count"] for p in paths) == [10.0, 10.0, 80.0]


def test_the_confusion_error_model():
    err = confusion_error(C, labels=[0, 1])
    assert abs(err(0, 0) - 0.9) < 1e-15
    assert abs(err(1, 0) - 0.2) < 1e-15
    assert abs(err(0, 0) + err(0, 1) - 1.0) < 1e-15


def test_figure_2_by_hand():
    err = confusion_error(C, labels=[0, 1])
    got = map_invert(lambda x: tree_predict(TREE, x), 1, {1: 1}, [0, 1],
                     err, PRIORS, sensitive=0)
    for v in (0, 1):
        want = err(1, tree_predict(TREE, [v, 1])) * PRIORS[0][v] * \
            PRIORS[1][1]
        assert abs(got["scores"][v] - want) < 1e-15
    assert got["estimate"] == 0


def test_a_strong_prior_overrides_the_likelihood():
    err = confusion_error(C, labels=[0, 1])
    skew = {0: {0: 0.001, 1: 0.999}, 1: {0: 0.3, 1: 0.7}}
    got = map_invert(lambda x: tree_predict(TREE, x), 1, {1: 1}, [0, 1],
                     err, skew, sensitive=0)
    assert got["estimate"] == 1


def test_the_white_box_estimator_reads_the_counts():
    wb = wbwc_invert(TREE, {}, [0, 1], FLAT, sensitive=0)
    assert wb["estimate"] == 0
    assert wb["scores"][0] > wb["scores"][1]
    assert (wb["n_paths"], wb["N"]) == (3, 100.0)
    flip = {"feature": 0, "branches": {
        1: {"label": 0, "count": 90.0},
        0: {"feature": 1, "branches": {
            1: {"label": 1, "count": 5.0},
            0: {"label": 0, "count": 5.0}}}}}
    assert wbwc_invert(flip, {}, [0, 1], FLAT, sensitive=0)["estimate"] == 1


def test_the_marginals_cannot_separate_equal_predictions():
    err = confusion_error(C, labels=[0, 1])
    assert tree_predict(TREE, [0, 0]) == tree_predict(TREE, [1, 0])
    got = map_invert(lambda x: tree_predict(TREE, x), 0, {1: 0}, [0, 1],
                     err, FLAT, sensitive=0)
    assert abs(got["scores"][0] - got["scores"][1]) < 1e-12


def _targets():
    out = []
    for x1 in (0, 1):
        for x2 in (0, 1):
            n = 40 if (x1, x2) == (0, 1) else 5
            for _ in range(n):
                out.append({"known": {1: x2},
                            "label": tree_predict(TREE, [x1, x2]),
                            "truth": x1})
    return out


def test_both_modes_run_and_score():
    t = _targets()
    bb = attrInf(TREE, t, PRIORS, confusion=C, labels=[0, 1],
                 mode="blackbox")
    wb = attrInf(TREE, t, FLAT, mode="whitebox")
    assert bb["mode"] == "blackbox" and wb["mode"] == "whitebox"
    assert len(bb["guesses"]) == len(t) == len(wb["guesses"])
    assert bb["accuracy"] is not None and wb["accuracy"] is not None
    assert "Figure 2" in bb["method"]
    assert "Equation 1" in wb["method"]


def test_validation():
    err = confusion_error(C, labels=[0, 1])
    t = _targets()
    for call in (lambda: tree_predict(TREE, [2, 1]),
                 lambda: tree_predict(TREE, [0]),
                 lambda: tree_predict({"nonsense": 1}, [0, 0]),
                 lambda: confusion_error([[1.0, 2.0]]),
                 lambda: confusion_error([[-1.0, 2.0], [1.0, 1.0]]),
                 lambda: confusion_error(C, labels=[0]),
                 lambda: map_invert(lambda x: 0, 1, {}, [], err, PRIORS),
                 lambda: wbwc_invert(TREE, {}, [], FLAT),
                 lambda: wbwc_invert(
                     {"feature": 0, "branches": {0: {"label": 0},
                                                 1: {"label": 1}}},
                     {}, [0, 1], FLAT),
                 lambda: attrInf(TREE, t, PRIORS, mode="greybox"),
                 lambda: attrInf(TREE, t, PRIORS, mode="blackbox"),
                 lambda: attrInf(TREE, t, {}, mode="whitebox",
                                 candidates=[])):
        try:
            call()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_alias():
    assert attribute_inference is attrInf
