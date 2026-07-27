"""Tests for ganls.gan_loss."""

import numpy as np
import pytest

from morie.fn.ganls import gan_loss


def test_ganls_at_the_uninformative_discriminator():
    """With D = 1/2 everywhere the discriminator has learned nothing, and the
    minimax value is the familiar -2 log 2 = -1.386 (Goodfellow et al. 2014)."""
    half = np.full(64, 0.5)
    r = gan_loss(half, half, kind="minimax")
    assert float(r["d_loss"]) == pytest.approx(2 * np.log(2), rel=1e-9)
    assert float(r["v"]) == pytest.approx(-2 * np.log(2), rel=1e-9)


def test_ganls_a_perfect_discriminator_beats_a_useless_one():
    """D_real -> 1 and D_fake -> 0 is the discriminator winning, so its loss
    must be strictly lower than at D = 1/2."""
    good = gan_loss(np.full(64, 0.99), np.full(64, 0.01))
    useless = gan_loss(np.full(64, 0.5), np.full(64, 0.5))
    assert float(good["d_loss"]) < float(useless["d_loss"])


def test_ganls_nonsaturating_differs_from_minimax_on_the_generator():
    """The non-saturating trick changes only the generator's objective: it uses
    -log D(G(z)) instead of log(1 - D(G(z))). The discriminator loss is shared."""
    real, fake = np.full(32, 0.8), np.full(32, 0.2)
    mm = gan_loss(real, fake, kind="minimax")
    ns = gan_loss(real, fake, kind="nonsaturating")
    assert float(mm["d_loss"]) == pytest.approx(float(ns["d_loss"]), rel=1e-12)
    assert float(mm["g_loss"]) != pytest.approx(float(ns["g_loss"]), rel=1e-6)
    assert float(ns["g_loss"]) == pytest.approx(-np.mean(np.log(fake)), rel=1e-9)


def test_ganls_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        gan_loss(np.full(4, 0.5), np.full(4, 0.5), kind="wasserstein")
