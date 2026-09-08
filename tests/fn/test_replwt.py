"""Replicate weights for survey variance estimation."""
import importlib

import pytest

R = importlib.import_module("morie.fn.replwt")

Y = [12.0, 20.0, 7.0, 9.5, 31.0, 24.0, 5.0, 6.5, 18.0, 11.0,
     40.0, 33.0]
W = [4.0, 4.0, 2.5, 2.5, 6.0, 6.0, 3.0, 3.0, 5.0, 5.0, 1.5, 1.5]
STR = ["h%d" % (i // 2) for i in range(12)]
D = R.design(W, STR, list(range(12)))
# two PSUs per stratum: V = sum_h (y_h1 - y_h2)^2
CLOSED = sum((W[2 * h] * Y[2 * h] - W[2 * h + 1] * Y[2 * h + 1]) ** 2
             for h in range(6))

SRS = [3.0, 8.0, 1.0, 6.0, 9.0, 4.0, 7.0, 2.0]
DSRS = R.design([1.0] * len(SRS), None, list(range(len(SRS))))


def total(w):
    return sum(wi * v for wi, v in zip(w, Y))


def mean(w):
    return sum(wi * v for wi, v in zip(w, SRS)) / sum(w)


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16])
def test_hadamard_rows_are_orthogonal(k):
    H = R.hadamard(k)
    for i in range(k):
        for j in range(k):
            assert sum(H[i][t] * H[j][t] for t in range(k)) \
                == (k if i == j else 0)
    assert all(v in (1, -1) for row in H for v in row)


def test_brr_reproduces_the_stratified_variance_exactly():
    v = R.replicate_variance(total, D, R.brr_weights(D))
    assert v["variance"] == pytest.approx(CLOSED, abs=1e-9)


def test_brr_uses_the_next_power_of_two():
    assert R.brr_weights(D)["n_replicates"] == 8


def test_fay_at_zero_is_brr():
    a, b = R.brr_weights(D, 0.0), R.brr_weights(D)
    assert a["weights"] == b["weights"]
    assert a["scale"] == b["scale"]


@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7])
def test_fay_recovers_the_same_variance(rho):
    v = R.replicate_variance(total, D, R.brr_weights(D, rho))
    assert v["variance"] == pytest.approx(CLOSED, abs=1e-9)


def test_fay_never_zeroes_a_unit():
    assert all(x > 0 for x in R.brr_weights(D, 0.3)["weights"][0])
    assert any(x == 0 for x in R.brr_weights(D)["weights"][0])


def test_jk1_on_a_mean_is_s_squared_over_n():
    n = len(SRS)
    mu = sum(SRS) / n
    s2 = sum((y - mu) ** 2 for y in SRS) / (n - 1)
    v = R.replicate_variance(mean, DSRS,
                             R.jackknife_weights(DSRS, "jk1"))
    assert v["variance"] == pytest.approx(s2 / n, abs=1e-12)


def test_jkn_recovers_the_stratified_variance():
    v = R.replicate_variance(total, D, R.jackknife_weights(D, "jkn"))
    assert v["variance"] == pytest.approx(CLOSED, abs=1e-9)


def test_jkn_drops_one_psu_per_replicate():
    jk = R.jackknife_weights(D, "jkn")
    assert len(jk["dropped"]) == 12
    for w, k in zip(jk["weights"], jk["dropped"]):
        assert all(w[i] == 0.0 for i in D["psu_units"][k])


@pytest.mark.parametrize("rep_of", ["brr", "jkn", "bootstrap"])
def test_replicates_preserve_the_total_weight(rep_of):
    rep = {"brr": lambda: R.brr_weights(D),
           "jkn": lambda: R.jackknife_weights(D, "jkn"),
           "bootstrap": lambda: R.bootstrap_weights(D, 40, 7)}[rep_of]()
    tot = sum(D["weights"])
    for w in rep["weights"]:
        assert sum(w) == pytest.approx(tot, abs=1e-9)
        assert all(x >= -1e-12 for x in w)


def test_the_bootstrap_is_reproducible():
    assert R.bootstrap_weights(D, 30, 11)["weights"] \
        == R.bootstrap_weights(D, 30, 11)["weights"]
    assert R.bootstrap_weights(D, 30, 11)["weights"] \
        != R.bootstrap_weights(D, 30, 12)["weights"]


def test_the_bootstrap_variance_lands_near_the_exact_one():
    v = R.replicate_variance(total, D, R.bootstrap_weights(D, 4000, 5))
    assert abs(v["variance"] - CLOSED) < 0.10 * CLOSED


def test_the_estimator_is_a_black_box():
    def ratio(w):
        return (sum(wi * y for wi, y in zip(w, Y)) / sum(w))
    v = R.replicate_variance(ratio, D, R.brr_weights(D))
    assert v["variance"] > 0
    assert v["std_error"] == pytest.approx(v["variance"] ** 0.5)


def test_a_constant_estimator_has_no_variance():
    v = R.replicate_variance(lambda w: 3.0, D, R.brr_weights(D))
    assert v["variance"] == pytest.approx(0.0, abs=1e-15)
    assert v["theta"] == 3.0


def test_the_entry_point_dispatches():
    for m in ("jk1", "jkn", "brr", "bootstrap"):
        r = R.replicate_weights(D, m, R=20)
        assert r["n_replicates"] == len(r["weights"]) >= 2
    assert R.replicate_weights(D, "fay", fay=0.5)["fay"] == 0.5


@pytest.mark.parametrize("call", [
    lambda: R.design([1.0]),
    lambda: R.design([1.0, -2.0]),
    lambda: R.design([1.0, 1.0], ["a", "b"], [1, 2]),
    lambda: R.design([1.0, 1.0], ["a"], [1, 2]),
    lambda: R.hadamard(6),
    lambda: R.hadamard(0),
    lambda: R.brr_weights(DSRS),
    lambda: R.brr_weights(D, 1.0),
    lambda: R.jackknife_weights(D, "jk9"),
    lambda: R.bootstrap_weights(D, 1),
    lambda: R.replicate_weights(D, "fay", fay=0.0),
    lambda: R.replicate_weights(D, "linearisation"),
])
def test_bad_input_is_refused(call):
    with pytest.raises(ValueError):
        call()
