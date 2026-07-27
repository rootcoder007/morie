"""Tests for posab.positional_encoding_abs."""

import numpy as np
import pytest

from morie.fn.posab import positional_encoding_abs


def test_posab_matches_the_closed_form():
    """PE is fully determined by Vaswani et al. (2017) eq. 1, so check it exactly."""
    seq_len, d_model, base = 6, 8, 10000.0
    pe = np.asarray(positional_encoding_abs(seq_len, d_model, base=base)["PE"], dtype=float)
    assert pe.shape == (seq_len, d_model)

    want = np.empty((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(d_model // 2):
            ang = pos / base ** (2 * i / d_model)
            want[pos, 2 * i] = np.sin(ang)
            want[pos, 2 * i + 1] = np.cos(ang)
    np.testing.assert_allclose(pe, want, atol=1e-12)


def test_posab_row_zero_is_sin0_cos0_and_norms_are_constant():
    pe = np.asarray(positional_encoding_abs(12, 16)["PE"], dtype=float)
    # Position 0 has every angle 0, so the row alternates sin(0), cos(0).
    np.testing.assert_allclose(pe[0, 0::2], 0.0, atol=1e-12)
    np.testing.assert_allclose(pe[0, 1::2], 1.0, atol=1e-12)
    # sin^2 + cos^2 = 1 per pair, so every row has the same norm.
    norms = np.linalg.norm(pe, axis=1)
    np.testing.assert_allclose(norms, np.sqrt(pe.shape[1] / 2), atol=1e-12)


def test_posab_rejects_nonpositive_dimensions():
    with pytest.raises(ValueError, match="must be > 0"):
        positional_encoding_abs(0, 8)
    with pytest.raises(ValueError, match="must be > 0"):
        positional_encoding_abs(4, 0)
