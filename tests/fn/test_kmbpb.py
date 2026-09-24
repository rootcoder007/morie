"""Verification tests for kmbpb.kamath_bits_per_byte.

Kamath, Keenan, Somers and Sorenson (2024), chapter 8, with the
standard definition of Gao et al. (2020). Bits per byte is the mean
negative log-probability per token, converted to bits and divided by
the bytes each token covers, so the expected values here are exact.
"""

import math

import pytest

from morie.fn.kmbpb import kamath_bits_per_byte


def test_two_coin_flip_tokens_cost_exactly_one_bit_per_byte():
    # each token has probability 1/2, so it costs one bit; at one byte
    # per token that is one bit per byte
    log_probs = [math.log(0.5), math.log(0.5)]
    res = kamath_bits_per_byte(log_probs, bytes_per_token=1.0)
    assert res["bits_per_byte"] == pytest.approx(1.0, rel=1e-12)
    assert res["bits_per_token"] == pytest.approx(1.0, rel=1e-12)
    assert res["cross_entropy"] == pytest.approx(math.log(2.0), rel=1e-12)
    assert res["perplexity"] == pytest.approx(2.0, rel=1e-12)


def test_bits_per_byte_scales_inversely_with_bytes_per_token():
    log_probs = [math.log(0.5)] * 4
    one = kamath_bits_per_byte(log_probs, bytes_per_token=1.0)["bits_per_byte"]
    two = kamath_bits_per_byte(log_probs, bytes_per_token=2.0)["bits_per_byte"]
    assert two == pytest.approx(one / 2.0, rel=1e-12)


def test_a_byte_total_and_a_per_token_rate_agree():
    log_probs = [math.log(0.25), math.log(0.5), math.log(0.125)]
    by_total = kamath_bits_per_byte(log_probs, n_bytes=6.0)
    by_rate = kamath_bits_per_byte(log_probs, bytes_per_token=2.0)
    assert by_total["bits_per_byte"] == pytest.approx(
        by_rate["bits_per_byte"], rel=1e-12)


def test_cross_entropy_is_the_mean_negative_log_probability():
    log_probs = [math.log(0.9), math.log(0.1), math.log(0.5)]
    res = kamath_bits_per_byte(log_probs, bytes_per_token=1.0)
    expected = -sum(log_probs) / len(log_probs)
    assert res["cross_entropy"] == pytest.approx(expected, rel=1e-12)
    assert res["bits_per_token"] == pytest.approx(expected / math.log(2.0), rel=1e-12)
    assert res["perplexity"] == pytest.approx(math.exp(expected), rel=1e-12)


def test_base_two_input_is_read_as_bits():
    # the same distribution given in bits must give the same answer
    in_nats = kamath_bits_per_byte([math.log(0.5)] * 3, bytes_per_token=1.0)
    in_bits = kamath_bits_per_byte([-1.0] * 3, bytes_per_token=1.0, base="2")
    assert in_bits["bits_per_byte"] == pytest.approx(
        in_nats["bits_per_byte"], rel=1e-12)


def test_a_positive_log_probability_is_refused():
    with pytest.raises(ValueError):
        kamath_bits_per_byte([0.5], bytes_per_token=1.0)


def test_the_byte_count_cannot_be_omitted():
    # without it the measure is undefined: the tokens cover unknown text
    with pytest.raises(ValueError):
        kamath_bits_per_byte([math.log(0.5)])


def test_an_empty_sequence_is_refused():
    with pytest.raises(ValueError):
        kamath_bits_per_byte([], bytes_per_token=1.0)


# --- appended: the module's own worked example as a gate -----------
# The docstring carries the printed value from the source the module
# cites. Executing it here makes that value a test-suite gate, on top
# of whatever the tests above already check.

import doctest as _doctest

import morie.fn.kmbpb as _doctest_module


def test_every_printed_value_in_the_worked_example_reproduces():
    res = _doctest.testmod(
        _doctest_module, verbose=False, report=False,
        optionflags=_doctest.NORMALIZE_WHITESPACE | _doctest.ELLIPSIS)
    assert res.attempted > 0
    assert res.failed == 0
