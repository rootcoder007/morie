"""Verification tests for km130.

Kamath, Keenan, Somers and Sorenson (2024), ch 9, the input-alignment loss, eq. 9.2. Expected values are
recomputed in the test body and the docstring's own worked value is
asserted too.
"""

import math

import pytest

from morie.fn.km130 import kamath_ch9_input_alignment_loss


def _mse(y, tgt):
    flat_y = [v for row in y for v in row]
    flat_t = [v for row in tgt for v in row]
    return sum((a - b) ** 2 for a, b in zip(flat_y, flat_t)) / len(flat_t)


def _add(p, f):
    return [[a + b for a, b in zip(pr, fr)] for pr, fr in zip(p, f)]


def test_the_alignment_loss_picks_the_candidate_prompt_that_fits_the_text():
    res = kamath_ch9_input_alignment_loss([[[0.0]], [[1.0]]], [[1.0]], [[1.0]],
               llm=_add, loss_fn=_mse)
    # candidate 0 gives [[1.0]] against a target of [[1.0]]
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)
    assert res["argmin"] == 0


def test_the_losses_of_every_candidate_are_reported():
    res = kamath_ch9_input_alignment_loss([[[0.0]], [[1.0]]], [[1.0]], [[1.0]],
               llm=_add, loss_fn=_mse)
    assert list(res["losses"]) == pytest.approx([0.0, 1.0], abs=1e-15)


def test_the_worse_candidate_wins_when_the_target_moves():
    res = kamath_ch9_input_alignment_loss([[[0.0]], [[1.0]]], [[1.0]], [[2.0]],
               llm=_add, loss_fn=_mse)
    assert res["argmin"] == 1
    assert res["estimate"] == pytest.approx(0.0, abs=1e-15)
