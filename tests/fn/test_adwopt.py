"""Tests for adwopt.adamw_step (AdamW single step)."""

import math

from morie.fn.adwopt import adamw_step


def test_adwopt_first_step_is_lr_signed():
    # from zero state, mhat/sqrt(vhat) = sign(g), theta = 0 -> each
    # update component has magnitude lr
    lr = 1e-3
    result = adamw_step([1.0, -2.0], lr=lr)
    assert result["t"] == 1
    assert result["step_norm"] == math.sqrt(2) * lr \
        or abs(result["step_norm"] - math.sqrt(2) * lr) < 1e-9
    mags = [abs(u) for u in result["update"]]
    assert all(abs(m - lr) < 1e-6 for m in mags)


def test_adwopt_state_advances():
    r1 = adamw_step([1.0], lr=1e-3)
    r2 = adamw_step([1.0], lr=1e-3, state=r1["state"],
                    theta=r1.get("theta"))
    assert r2["t"] == 2
    assert r2["step_norm"] > 0
