"""Tests for alffap.alphafold_fape_loss."""

import math

from morie.fn import _array_core as np
from morie.fn import _frame_core as pd

from morie.fn.alffap import alphafold_fape_loss


def _make_frames(rng, nframes):
    """Build a list of [R, t] frames, where R is a 3x3 rotation-ish
    matrix and t is a length-3 translation vector."""
    frames = []
    for _ in range(nframes):
        R = rng.normal(0.0, 1.0, (3, 3))
        t = rng.normal(0.0, 1.0, 3)
        frames.append([R, t])
    return frames


def _make_points(rng, natoms):
    """Build a list of length-3 atom position vectors."""
    return [rng.normal(0.0, 1.0, 3) for _ in range(natoms)]


def test_alffap_basic():
    """Test basic functionality.

    With predicted frames == true frames and predicted points == true
    points, every pair difference is the zero vector, so line 3 reduces
    to sqrt(eps).  Line 4 then clamps each contribution, and the loss
    is sqrt(eps) / Z for nf * na pairs.
    """
    rng = np.random.default_rng(42)

    nframes = 3
    natoms = 4
    frames = _make_frames(rng, nframes)
    points = _make_points(rng, natoms)

    result = alphafold_fape_loss(
        frames, points, frames, points, Z=10.0, dclamp=10.0, eps=1e-4
    )

    # The function returns a RichResult; support both dict-style and
    # attribute-style access so the test is robust to either wrapper.
    payload = result.payload if hasattr(result, "payload") else result
    assert isinstance(payload, dict)

    assert "estimate" in payload
    assert "d" in payload
    assert "nframes" in payload
    assert "natoms" in payload
    assert "method" in payload

    # Independent computation of the documented closed form:
    # every dij = sqrt(eps), clamp(10 > eps) leaves them at sqrt(eps),
    # so loss = sqrt(eps) / Z.
    expected_estimate = math.sqrt(1e-4) / 10.0
    assert abs(payload["estimate"] - expected_estimate) < 1e-12

    # Shape bookkeeping.
    assert payload["nframes"] == nframes
    assert payload["natoms"] == natoms
    assert len(payload["d"]) == nframes
    for row in payload["d"]:
        assert len(row) == natoms


def test_alffap_edge():
    """Test edge cases.

    Pushing every predicted atom far from the true atom under every
    predicted frame drives every dij above dclamp, so line 4 saturates
    to dclamp and the loss becomes dclamp / Z regardless of how far
    apart the points actually are (the closed form anchored in the
    docstring's Notes).
    """
    rng = np.random.default_rng(7)

    nframes = 2
    natoms = 3
    frames_pred = _make_frames(rng, nframes)
    frames_true = _make_frames(rng, nframes)

    # Truth near the origin; predictions offset by a large translation
    # of 1000 angstroms along x, so every pair is far beyond dclamp.
    x_true = [rng.normal(0.0, 0.01, 3) for _ in range(natoms)]
    x_pred = [p + 1000.0 for p in x_true]

    result = alphafold_fape_loss(
        frames_pred, x_pred,
        frames_true, x_true,
        Z=10.0, dclamp=10.0, eps=1e-4,
    )

    payload = result.payload if hasattr(result, "payload") else result
    assert isinstance(payload, dict)

    expected_estimate = 10.0 / 10.0  # dclamp / Z
    assert abs(payload["estimate"] - expected_estimate) < 1e-9
