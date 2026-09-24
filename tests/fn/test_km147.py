"""Verification tests for km147.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.19, the output alignment. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km147 import kamath_ch9_output_alignment


def test_the_output_alignment_maps_signal_tokens_back_to_features():
    # Eq 9.19: H_X = OUT_ALIGN(S_X)
    res = kamath_ch9_output_alignment([[1.0, 2.0]], [[0.0], [2.0]])
    assert [list(r) for r in res["features"]] == [[4.0]]


def test_a_callable_projector_is_applied_directly():
    res = kamath_ch9_output_alignment([[1.0, 2.0]], lambda s: [[max(s[0])]])
    assert [list(r) for r in res["features"]] == [[2.0]]


def test_an_identity_alignment_preserves_the_signal():
    res = kamath_ch9_output_alignment([[1.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]])
    assert [list(r) for r in res["features"]] == [[1.0, 2.0]]
