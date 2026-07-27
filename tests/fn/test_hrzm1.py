"""Tests for hrzm1.horowitz_mixture_model."""

import numpy as np
import pytest

from morie.fn.hrzm1 import horowitz_mixture_model


def test_hrzm1_separates_two_well_split_components():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(-3, 0.5, 400), rng.normal(3, 0.5, 600)])
    r = horowitz_mixture_model(y, k=2)
    means = np.sort(np.asarray(r["estimate"]["mu"], dtype=float).ravel())
    assert means[0] == pytest.approx(-3.0, abs=0.3)
    assert means[-1] == pytest.approx(3.0, abs=0.3)


def test_hrzm1_weights_reflect_the_mixing_proportion():
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(-4, 0.5, 300), rng.normal(4, 0.5, 700)])
    r = horowitz_mixture_model(y, k=2)
    w = np.sort(np.asarray(r["estimate"]["pi"], dtype=float).ravel())
    assert w[0] == pytest.approx(0.3, abs=0.08)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
