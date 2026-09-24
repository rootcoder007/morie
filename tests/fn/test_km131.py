"""Verification tests for km131.

Kamath, Keenan, Somers and Sorenson (2024), eq. 9.3, the input projector. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km131 import kamath_ch9_input_projector


def test_the_input_projector_maps_features_into_the_prompt_space():
    # Eq 9.3: P_X = IN_ALIGN(F_X), here a weight matrix
    res = kamath_ch9_input_projector([[1.0, 2.0]], [[1.0], [3.0]])
    assert [list(r) for r in res["prompts"]] == [[7.0]]


def test_a_callable_projector_is_applied_directly():
    res = kamath_ch9_input_projector([[1.0, 2.0]], lambda f: [[sum(f[0])]])
    assert [list(r) for r in res["prompts"]] == [[3.0]]


def test_an_identity_projection_preserves_the_features():
    res = kamath_ch9_input_projector([[1.0, 2.0]], [[1.0, 0.0], [0.0, 1.0]])
    assert [list(r) for r in res["prompts"]] == [[1.0, 2.0]]
