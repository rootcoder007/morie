"""Tests for causivft.causal_iv_first_stage (Stock-Wright-Yogo first-stage F).

Anchored on the R-squared form of the same statistic,
(R2/(1-R2)) * (n-L-1)/L, computed here directly from the first-stage
fit rather than through causal_iv_first_stage.
"""

import pytest

from morie.fn import _array_core as np
from morie.fn.causivft import causal_iv_first_stage

N = 40
_I = list(range(N))
Z0 = [0.2 + 0.09 * i + 0.25 * ((i * i) % 7) for i in _I]
Z1 = [1.1 - 0.05 * i + 0.4 * ((i * i) % 7) for i in _I]
D = [0.8 * Z0[i] + 0.6 * Z1[i] + 0.3 * ((i * 2) % 3) for i in _I]
Z = np.column_stack([np.asarray(Z0), np.asarray(Z1)])


def test_first_stage_f_equals_r_squared_form():
    d = np.asarray(D, dtype=float)
    Dz = np.column_stack([np.ones(N), Z])
    b, *_ = np.linalg.lstsq(Dz, d, rcond=None)
    r = d - Dz @ b
    r2 = 1.0 - float(r @ r) / float(((d - np.mean(d)) ** 2).sum())
    expected = (r2 / (1.0 - r2)) * (N - 2 - 1) / 2

    res = causal_iv_first_stage(D, Z)
    assert res["df1"] == 2
    assert res["df2"] == N - 3
    assert res["statistic"] == pytest.approx(expected, rel=1e-9)


def test_irrelevant_instruments_are_flagged_weak():
    noise = np.column_stack(
        [np.asarray([float((i * 7) % 5) for i in _I]), np.asarray([float((i * 11) % 3) for i in _I])]
    )
    res = causal_iv_first_stage([float((i * 13) % 17) for i in _I], noise)
    assert res["weak"] is True
    assert res["statistic"] < 10.0


def test_strong_instruments_are_not_flagged_weak():
    res = causal_iv_first_stage(D, Z)
    assert res["weak"] is False
    assert res["statistic"] > 10.0
