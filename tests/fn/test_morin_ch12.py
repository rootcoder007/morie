"""Morin (2016) chapters 1-2: tests anchored on the book's worked numbers."""
import math

import numpy as np
import pytest

from morie.fn import _morin

P = "david_j_morin_probability_for_the_enthusiastic_beginner"


def front(suffix):
    import importlib
    mod = importlib.import_module(f"morie.fn.{P}{suffix}")
    ch, e = suffix.split("e")
    return getattr(mod, f"{P}_chapter_{ch}_equation_{e}")


# ------------------------------------------------------------- chapter 1

def test_factorials_match_eq_1_1_table():
    # book: 1!..6! = 1, 2, 6, 24, 120, 720; 10! = 3,628,800
    got = [front("1e1")(n)["factorial"] for n in range(1, 7)]
    assert got == [1, 2, 6, 24, 120, 720]
    assert front("1e1")(10)["factorial"] == 3_628_800


def test_permutations_eq_1_2_table():
    # book eq (1.2): P1..P5 = 1, 2, 6, 24, 120
    got = [front("1e3")(n)["permutations"] for n in range(1, 6)]
    assert got == [1, 2, 6, 24, 120]


def test_partial_permutations_product_and_factorial_forms():
    r = front("1e5")(10, 3)
    assert r["partial_permutations"] == 10 * 9 * 8
    # n = N reduces to N! (book note below eq (1.6))
    assert front("1e5")(6, 6)["partial_permutations"] == math.factorial(6)
    with pytest.raises(ValueError):
        front("1e5")(3, 5)


def test_binomial_theorem_eq_1_21():
    r = front("1e21")(1.0, 1.0, 8)
    # a = b = 1 gives sum C(8,k) = 2^8 (book's proof of eq (1.19))
    assert r["sum"] == pytest.approx(256.0)
    r2 = front("1e21")(2.0, -0.5, 7)
    assert r2["sum"] == pytest.approx(1.5 ** 7, rel=1e-12)


def test_hockey_stick_eq_1_29():
    for n, k in [(6, 2), (10, 4), (12, 1)]:
        r = front("1e29")(n, k)
        assert r["identity_holds"] and r["sum"] == math.comb(n, k)


def test_committee_counts_eqs_1_35():
    # book: 10 people into committees of 3, 2, 5 -> 10!/(3!2!5!) = 2520
    assert front("1e35")([3, 2, 5])["assignments"] == 2520
    # book: same committees from 16 people -> 16!/(3!2!5!6!) ~ 20 million
    r = front("1e35")([3, 2, 5], 16)
    assert r["assignments"] == math.factorial(16) // (
        math.factorial(3) * math.factorial(2) * math.factorial(5) * math.factorial(6))
    assert 19_000_000 < r["assignments"] < 21_000_000


def test_multinomial_coefficient_eq_1_37_reduces_to_binomial():
    assert front("1e37")([4, 6])["coefficient"] == math.comb(10, 4)
    assert front("1e37")([2, 3, 4])["coefficient"] == 1260


def test_stars_and_bars_eq_1_57():
    # book eq (1.16)/(1.57); N = 1 gives 1 for any n
    assert front("1e57")(5, 1)["count"] == 1
    assert front("1e57")(2, 3)["count"] == 6
    assert front("1e57")(3, 10)["count"] == math.comb(12, 9)


def test_sd_of_dice_sum_book_page_253():
    # book: sd of one die sqrt(2.92) = 1.71, sum of 10 dice -> sqrt(10)(1.71) = 5.4
    r = front("1e71")(math.sqrt(2.92), 10)
    assert r["sd_sum"] == pytest.approx(5.4, abs=0.01)


# ------------------------------------------------------------- chapter 2

def test_and_rule_eqs_2_2_to_2_4():
    # dice: P(2 and 5) = 1/36; cards: P(king and heart) = 1/52
    assert front("2e2")(1 / 6, 1 / 6)["p_and"] == pytest.approx(1 / 36)
    assert front("2e3")([1 / 13, 1 / 4])["p_and"] == pytest.approx(1 / 52)


def test_chain_rule_eq_2_9_consistency():
    # red/blue balls example: P(Red1 and Blue2) = (2/5)(3/4) = 3/10
    r = front("2e9")(2 / 5, 3 / 4, 3 / 5, 1 / 2)
    assert r["p_and"] == pytest.approx(3 / 10)
    with pytest.raises(ValueError):
        front("2e9")(0.5, 0.5, 0.9, 0.9)


def test_or_rules_eqs_2_14_2_21():
    # diamond or heart = 1/2; king or heart = 4/13
    assert front("2e14")([1 / 4, 1 / 4])["p_or"] == pytest.approx(1 / 2)
    assert front("2e21")(1 / 13, 1 / 4, 1 / 52)["p_or"] == pytest.approx(4 / 13)


def test_classification_eq_2_24():
    # A = roll a 2, B = roll even on same die: dependent, non-exclusive
    r = front("2e24")(1 / 6, 1 / 2, 1 / 6)
    assert not r["independent"] and not r["exclusive"]
    r2 = front("2e24")(1 / 6, 1 / 6, 1 / 36)
    assert r2["independent"] and not r2["exclusive"]


def test_total_probability_eq_2_29():
    # book: P(Mmid) = (1/3)(0) + (2/3)(1/2) = 1/3
    assert front("2e29")([1 / 3, 2 / 3], [0.0, 0.5])["p_event"] == pytest.approx(1 / 3)


def test_suit_full_house_eq_2_41():
    r = front("2e41")()
    assert r["favorable"] == 267_696
    assert r["total"] == 2_598_960
    assert r["probability"] == pytest.approx(0.103, abs=5e-4)


def test_at_most_two_suits_eqs_2_42_2_43():
    r = front("2e43")()
    assert r["favorable"] == 384_384  # book: 394,680 - 10,296
    assert r["probability"] == pytest.approx(0.148, abs=5e-4)


def test_conditionals_eqs_2_48_2_49():
    assert front("2e48")(0.1, 0.4)["p_b_given_a"] == pytest.approx(0.25)
    # boy/girl part (a): P(2 boys | at least 1 boy) = (1/4)/(3/4) = 1/3
    assert front("2e49")(1 / 4, 3 / 4)["p_b_given_a"] == pytest.approx(1 / 3)


def test_bayes_simple_eq_2_51_consistency():
    p_a, p_z_a, p_z = 0.02, 0.95, 0.02 * 0.95 + 0.98 * 0.10
    assert front("2e51")(p_z_a, p_a, p_z)["posterior"] == pytest.approx(
        front("2e58")()["posterior"])


def test_evidence_eq_2_55():
    assert front("2e55")(0.02, 0.95, 0.10)["p_z"] == pytest.approx(0.117)


def test_false_positive_anchors_eqs_2_58_2_61_2_62():
    # book: 0.16 with 2% prevalence, 0.86 with 40%
    assert front("2e58")()["posterior"] == pytest.approx(19 / 117, rel=1e-12)
    assert round(front("2e61")()["posterior"], 2) == 0.16
    assert round(front("2e62")()["posterior"], 2) == 0.86
    assert front("2e59")(0.02, 0.95, 0.10)["posterior"] == pytest.approx(19 / 117)


def test_exact_vs_stirling_eqs_2_65_2_66():
    # book: exact 0.07959, Stirling 0.07979 at n = 50
    assert front("2e65")(50)["probability"] == pytest.approx(0.07959, abs=5e-6)
    r = front("2e66")(50)
    assert r["approx"] == pytest.approx(0.07979, abs=5e-6)
    assert r["relative_error"] < 0.003
    # approximation improves with n
    assert front("2e66")(500)["relative_error"] < front("2e66")(50)["relative_error"]


def test_summary_rules_eqs_2_70_2_74():
    assert front("2e70")(0.3, 0.4)["p_and"] == pytest.approx(0.12)
    post = front("2e74")([0.02, 0.98], [0.95, 0.10])["posteriors"]
    assert post[0] == pytest.approx(19 / 117)
    assert sum(post) == pytest.approx(1.0)


def test_decomposition_eq_2_86():
    # book second solution: P(B) = (2/5)(1/10)... anchor 64% example:
    # P(B) = P(A)P(B|A) + P(~A)P(B|~A) with 16/25 = 64%
    assert front("2e86")(2 / 5, 1 / 10, 14 / 15)["p_b"] == pytest.approx(
        (2 / 5) * (1 / 10) + (3 / 5) * (14 / 15))


def test_three_dice_eqs_2_92_to_2_96():
    # at least one 6 in three rolls: 3/6 - 3/36 + 1/216 = 91/216
    p = front("2e92")(1 / 6, 1 / 6, 1 / 6, 1 / 36, 1 / 36, 1 / 36, 1 / 216)["p_or"]
    assert p == pytest.approx(91 / 216, rel=1e-12)
    assert front("2e95")(1 / 6, 2)["p_intersection"] == pytest.approx(1 / 36)
    assert front("2e95")(1 / 6, 3)["p_intersection"] == pytest.approx(1 / 216)
    assert front("2e96")(1 / 6, 3)["p_at_least_one"] == pytest.approx(91 / 216)


def test_backend_validation():
    with pytest.raises(ValueError):
        _morin.prob_and_independent([0.5, 1.5])
    with pytest.raises(ValueError):
        _morin.total_probability([0.5, 0.4], [0.5, 0.5])  # priors don't sum to 1
    with pytest.raises(ValueError):
        _morin.bayes_explicit(0.5, 0.0, 0.0)
    with pytest.raises(ValueError):
        _morin.conditional_from_joint(0.6, 0.5)


def test_richresult_contract():
    r = front("2e58")()
    assert isinstance(r, dict)
    assert "posterior" in r
    assert str(r) != ""
