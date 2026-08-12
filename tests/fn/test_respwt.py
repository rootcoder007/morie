"""Tests for respwt (weighting-class nonresponse adjustment, Lohr 2010).

Replaces the generated stub, which imported a name the module never had.
"""

from morie.fn.respwt import respwt, response_weight_adjustment


def test_the_books_printed_constants():
    # Lohr Example 8.4 / Table 8.2: the 15-24 class has phi_hat = 0.6165
    # and so a weight factor of 1.622. Reconstructed here from the
    # response rate itself, which is what those two numbers encode.
    n, r = 10000, 6165
    w = [1.0] * n
    responded = [True] * r + [False] * (n - r)
    cls = ["15-24"] * n
    res = respwt(w, responded, cls)
    assert abs(res["phi_hat"]["15-24"] - 0.6165) < 1e-9
    assert abs(res["factors"]["15-24"] - 1.622) < 1e-3


def test_the_balance_identity_holds_exactly():
    # adjusted respondent weights in a class sum to the class's original
    # full-sample total -- the identity the module says it verifies
    w = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0]
    responded = [True, False, True, True, False, True]
    cls = ["a", "a", "a", "b", "b", "b"]
    res = respwt(w, responded, cls)
    assert abs(res["balance_error"]) < 1e-12
    for c in ("a", "b"):
        total = sum(w[i] for i in range(6) if cls[i] == c)
        adj = sum(res["adjusted"][i] for i in range(6)
                  if cls[i] == c and responded[i])
        assert abs(adj - total) < 1e-9


def test_full_response_leaves_the_weights_alone():
    w = [1.0, 2.0, 3.0]
    res = respwt(w, [True] * 3, ["x"] * 3)
    assert abs(res["factors"]["x"] - 1.0) < 1e-12
    for i in range(3):
        assert abs(res["adjusted"][i] - w[i]) < 1e-12


def test_nonrespondents_carry_no_weight():
    w = [1.0, 1.0, 1.0, 1.0]
    responded = [True, True, False, False]
    res = respwt(w, responded, ["x"] * 4)
    # a nonrespondent has no adjusted weight at all -- None, not a
    # zero that would quietly average into an estimate
    assert res["adjusted"][2] is None and res["adjusted"][3] is None
    assert abs(res["adjusted"][0] - 2.0) < 1e-12


def test_each_class_is_adjusted_on_its_own():
    w = [1.0] * 8
    responded = [True, True, True, False, True, False, False, False]
    cls = ["a"] * 4 + ["b"] * 4
    res = respwt(w, responded, cls)
    assert abs(res["phi_hat"]["a"] - 0.75) < 1e-12
    assert abs(res["phi_hat"]["b"] - 0.25) < 1e-12
    assert res["factors"]["b"] > res["factors"]["a"]


def test_alias():
    assert response_weight_adjustment is respwt
