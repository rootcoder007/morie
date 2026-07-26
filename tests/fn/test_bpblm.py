"""bpblm: bits-per-byte (Gao et al. 2020, The Pile).

    BPB = nll_nats / (ln 2 * n_bytes)
"""

import numpy as np
import pytest

from morie.fn.bpblm import bits_per_byte as bpb


def test_bpblm_matches_the_closed_form():
    nll = np.array([1.0, 2.0, 3.0])       # nats
    n_bytes = 4
    got = bpb(nll, n_bytes=n_bytes)
    assert got["nll_nats"] == pytest.approx(6.0)
    assert got["value"] == pytest.approx(6.0 / (np.log(2) * n_bytes))
    assert got["n_bytes"] == 4
    assert got["n_tokens"] == 3


def test_bpblm_one_bit_per_byte_is_ln2_nats_per_byte():
    """Calibration point: a model costing exactly ln(2) nats per byte scores
    exactly 1.0 bits per byte."""
    n_bytes = 100
    nll = np.full(n_bytes, np.log(2.0))
    assert bpb(nll, n_bytes=n_bytes)["value"] == pytest.approx(1.0)


def test_bpblm_a_perfect_model_scores_zero():
    assert bpb(np.zeros(10), n_bytes=10)["value"] == pytest.approx(0.0)


def test_bpblm_normalises_by_BYTES_not_by_tokens():
    """This is the whole point of BPB over per-token loss: it is comparable
    across tokenizers. The same total nats over twice the bytes halves it.
    """
    nll = np.full(8, 1.0)
    assert bpb(nll, n_bytes=20)["value"] == pytest.approx(
        2.0 * bpb(nll, n_bytes=40)["value"]
    )


def test_bpblm_is_tokenizer_invariant_at_fixed_total_loss():
    """Splitting the same total nats over more tokens must not change BPB --
    only the byte count matters."""
    coarse = np.array([4.0, 4.0])
    fine = np.full(8, 1.0)
    assert bpb(coarse, n_bytes=32)["value"] == pytest.approx(
        bpb(fine, n_bytes=32)["value"]
    )


def test_bpblm_rejects_a_non_positive_byte_count():
    with pytest.raises((ValueError, ZeroDivisionError)):
        bpb(np.ones(3), n_bytes=0)
