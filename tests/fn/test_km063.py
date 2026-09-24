"""Verification tests for km063.

Kamath, Keenan, Somers and Sorenson (2024), ch 4, the VeRA forward pass, eq. 4.10. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km063 import kamath_ch4_vera_forward


def test_the_vera_forward_pass_adds_the_scaled_random_projection():
    # h = W_0 x + Lambda_b B Lambda_d A x, with A and B frozen random
    W0 = [[1.0, 0.0], [0.0, 1.0]]
    res = kamath_ch4_vera_forward(W0, [3.0, 1.0], [2.0], [[1.0, 0.0]], [[1.0], [0.0]],
               [1.0, 2.0])
    # A x = [1]; Lambda_d A x = [2]; B [2] = [2, 0]; Lambda_b [2, 0] = [6, 0]
    assert list(res["h"]) == pytest.approx([1.0 + 6.0, 2.0 + 0.0], rel=1e-12)


def test_vera_trains_fewer_parameters_than_lora_on_the_same_shapes():
    # only the two scaling vectors are trainable, not A and B
    res = kamath_ch4_vera_forward([[1.0, 0.0], [0.0, 1.0]], [3.0, 1.0], [2.0],
               [[1.0, 0.0]], [[1.0], [0.0]], [1.0, 2.0])
    assert res["n_trainable"] == 2 + 1
    assert res["n_trainable_lora"] == 2 + 2
    assert res["n_trainable"] < res["n_trainable_lora"]


def test_zero_scaling_vectors_leave_the_frozen_weights_untouched():
    res = kamath_ch4_vera_forward([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], [2.0],
               [[1.0, 0.0]], [[1.0], [0.0]], [1.0, 2.0])
    assert list(res["h"]) == pytest.approx([1.0, 2.0], rel=1e-12)
