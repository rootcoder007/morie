"""Morin (2016) chapter 3: expectation and variance, book-anchored."""
import math

from morie.fn import _array_core as np
import pytest

from morie.fn import _morin

P = "david_j_morin_probability_for_the_enthusiastic_beginner"


def front(suffix):
    import importlib
    mod = importlib.import_module(f"morie.fn.{P}{suffix}")
    ch, e = suffix.split("e")
    return getattr(mod, f"{P}_chapter_{ch}_equation_{e}")


def test_independence_eq_3_9():
    indep = np.outer([0.3, 0.7], [0.4, 0.6])
    assert front("3e9")(indep)["independent"]
    dep = np.array([[0.5, 0.0], [0.0, 0.5]])
    assert not front("3e9")(dep)["independent"]


def test_convolution_book_example_eqs_3_11_3_12():
    # X fair on {1,2}, Y uniform on {1,2,3}: P(2..5) = 1/6, 2/6, 2/6, 1/6, E = 3.5
    r = front("3e11")([1, 2], [0.5, 0.5], [1, 2, 3], [1/3, 1/3, 1/3])
    assert r["values"] == [2.0, 3.0, 4.0, 5.0]
    assert r["probs"] == pytest.approx([1/6, 2/6, 2/6, 1/6])
    assert front("3e12")([1, 2], [0.5, 0.5], [1, 2, 3],
                         [1/3, 1/3, 1/3])["e_sum"] == pytest.approx(3.5)


def test_linearity_eqs_3_13_3_15():
    assert front("3e13")(2.0, 1.5, 3.0, 2.0, 1.0)["expectation"] == pytest.approx(10.0)
    assert front("3e15")(3.5, 10)["e_sum"] == pytest.approx(35.0)


def test_variance_anchors_eqs_3_19_to_3_22():
    # die: 35/12 = 2.9166..., book rounds to 2.92
    r = front("3e20")()
    assert r["variance"] == pytest.approx(35/12, rel=1e-12)
    assert round(r["variance"], 2) == 2.92
    assert front("3e21")()["variance"] == pytest.approx(0.25)
    assert front("3e22")(0.5)["variance"] == pytest.approx(0.25)
    v, mu = _morin.pmf_variance([1, 2, 3, 4, 5, 6], [1/6] * 6)
    assert front("3e19")([1, 2, 3, 4, 5, 6], [1/6] * 6)["variance"] == pytest.approx(v)


def test_variance_algebra_eqs_3_24_to_3_31():
    assert front("3e24")(3.0, 2.0)["var_aX"] == pytest.approx(18.0)
    assert front("3e25")(0.25, 0.25)["var_sum"] == pytest.approx(0.5)
    assert front("3e26")(1.0, 1.0, 0.5)["var_sum"] == pytest.approx(3.0)
    assert front("3e28")()["variance"] == pytest.approx(0.5)
    assert front("3e30")([1.0, 2.0, 3.0])["var_sum"] == pytest.approx(6.0)
    r = front("3e31")([1.0, 2.0, 3.0])
    assert r["var_sum"] == pytest.approx(6.0)
    assert r["partial_sums"] == pytest.approx([1.0, 3.0, 6.0])
    with pytest.raises(ValueError):
        front("3e26")(1.0, 1.0, 2.0)  # |cov| > sqrt(v1 v2)


def test_binomial_variance_eq_3_33():
    assert front("3e33")(100, 0.5)["variance"] == pytest.approx(25.0)
    # book after eq (3.33): flipping fair coin n times -> n/4
    assert front("3e33")(10, 0.5)["variance"] == pytest.approx(2.5)


def test_computational_form_eqs_3_34_3_35():
    vals, probs = [1, 2, 3, 4, 5, 6], [1/6] * 6
    assert front("3e34")(vals, probs)["variance"] == pytest.approx(35/12)
    assert front("3e35")(vals, probs)["variance"] == pytest.approx(35/12)


def test_dataset_variance_eqs_3_37_3_60_3_66_3_73():
    x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    assert front("3e37")(x)["variance"] == pytest.approx(4.0)
    assert front("3e60")(x)["variance"] == pytest.approx(4.0)
    r = front("3e66")(x)
    assert r["identity_error"] < 1e-12
    # sample variance uses n-1: 32/7
    assert front("3e73")(x)["sample_variance"] == pytest.approx(32/7)


def test_sd_forms_eqs_3_39_to_3_48():
    assert front("3e39")(4.0)["sd"] == pytest.approx(2.0)
    assert front("3e40")([0, 1], [0.5, 0.5])["sd"] == pytest.approx(0.5)
    assert front("3e41")(-3.0, 2.0)["sd_aX"] == pytest.approx(6.0)
    assert front("3e42")(3.0, 4.0)["sd_sum"] == pytest.approx(5.0)
    assert front("3e43")([3.0, 4.0])["sd_sum"] == pytest.approx(5.0)
    assert front("3e45")(1.0, 9)["sd_sum"] == pytest.approx(3.0)
    # book: p = 1/10 gives sqrt(pq) = 3/10
    assert front("3e46")(0.1)["sd"] == pytest.approx(0.3)
    assert front("3e47")(100, 0.5)["sd"] == pytest.approx(5.0)
    assert front("3e48")(100)["sd"] == pytest.approx(5.0)


def test_mean_sds_eqs_3_51_to_3_55():
    assert front("3e51")(64)["sd_tot"] == pytest.approx(4.0)
    assert front("3e52")(64)["sd_avg"] == pytest.approx(1/16)
    assert front("3e53")(2.0, 4)["sd_mean"] == pytest.approx(1.0)
    assert front("3e54")([1.0, 2.0, 3.0])["mean"] == pytest.approx(2.0)
    # equal sigmas reduce eq (3.55) to eq (3.53)
    assert front("3e55")([2.0, 2.0, 2.0, 2.0])["sd_avg"] == pytest.approx(
        front("3e53")(2.0, 4)["sd_mean"])
    assert front("3e55")([3.0, 4.0])["sd_avg"] == pytest.approx(2.5)


def test_dice_worked_block_eqs_3_56_to_3_58():
    # book: sqrt(10000 (1/6)(5/6)) = 37 (rounded); avg = 0.0037
    r56 = front("3e56")()
    assert r56["sd_tot"] == pytest.approx(math.sqrt(10000 * (1/6) * (5/6)), rel=1e-12)
    assert round(r56["sd_tot"]) == 37
    r57 = front("3e57")()
    assert round(r57["sd_avg"], 4) == 0.0037
    r58 = front("3e58")(10000, 1/6)
    assert round(r58["sd_single"], 2) == 0.37
    assert round(r58["sd_tot"]) == 37
    assert round(r58["sd_avg"], 4) == 0.0037


def test_summary_and_estimator_eqs_3_59_to_3_93():
    assert front("3e59")([0, 1], [0.5, 0.5])["variance"] == pytest.approx(0.25)
    assert front("3e70")(2.0, 3.0)["e_x2"] == pytest.approx(13.0)
    assert front("3e92")(2.0, 16)["var_mean"] == pytest.approx(0.25)
    r = front("3e93")(2.0, 16)
    assert r["sd_mean"] == pytest.approx(0.5)
    assert r["bounded"] and r["sd_mean"] <= 2.0


def test_sample_variance_unbiased_by_simulation():
    # E[s^2] = sigma^2; E[s-tilde^2] = (n-1)/n sigma^2 (book Sec 3.5)
    rng = np.random.default_rng(20260801)
    n, reps = 5, 4000
    s2 = np.empty(reps)
    st2 = np.empty(reps)
    for i in range(reps):
        x = rng.normal(0.0, 1.0, n)
        s2[i] = _morin.sample_variance(x)
        st2[i] = _morin.population_variance(x)
    # se of the mean of s^2 over reps ~ sqrt(2/(n-1))/sqrt(reps) ~ 0.011
    assert abs(s2.mean() - 1.0) < 5 * math.sqrt(2 / (n - 1) / reps)
    assert abs(st2.mean() - (n - 1) / n) < 5 * math.sqrt(2 / (n - 1) / reps)


def test_validation():
    with pytest.raises(ValueError):
        front("3e19")([1, 2], [0.6, 0.6])
    with pytest.raises(ValueError):
        front("3e73")([1.0])
    with pytest.raises(ValueError):
        front("3e92")(2.0, 0)
