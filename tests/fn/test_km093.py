"""Verification tests for km093.

Kamath, Keenan, Somers and Sorenson (2024), the HONEST hurtful-completion score. Expected values are
recomputed in the test body, and the value the docstring quotes is
asserted as well.
"""

import math

import pytest

from morie.fn.km093 import kamath_ch6_honest_score


def test_honest_is_the_hurtful_share_of_all_completions():
    # HONEST = hurtful completions / (|Yhat| . k)
    res = kamath_ch6_honest_score([["a", "bad"], ["ok", "fine"]], 2, hurtlex={"bad"})
    assert res["estimate"] == pytest.approx(0.25, rel=1e-12)
    assert res["n_hurtful"] == 1


def test_no_hurtful_completion_scores_zero():
    res = kamath_ch6_honest_score([["a", "b"]], 2, hurtlex={"zzz"})
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)


def test_every_completion_hurtful_scores_one():
    res = kamath_ch6_honest_score([["bad", "bad"]], 2, hurtlex={"bad"})
    assert res["estimate"] == pytest.approx(1.0, rel=1e-12)
