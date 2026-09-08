"""mdspl: classical (Torgerson) MDS.

Armstrong et al., *Analyzing Spatial Models of Choice and Judgment*,
section 3.1, printed p.68 -- verified against the PDF table of contents and
body text. The module previously cited "Armstrong Ch 7"; that book has six
chapters.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.mdspl import mds_spatial_map as mds


def _pdist(X):
    d = X[:, None, :] - X[None, :, :]
    return np.sqrt((d**2).sum(-1))


def test_mdspl_recovers_a_planted_square_exactly():
    """Four unit-square corners are exactly 2-D Euclidean, so classical MDS
    reproduces every pairwise distance and stress is 0."""
    X = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    r = mds(X, k=2)
    got = _pdist(np.asarray(r["coords"]))
    assert got == pytest.approx(_pdist(X), abs=1e-9)
    assert r["stress"] == pytest.approx(0.0, abs=1e-9)


def test_mdspl_is_invariant_to_rotation_and_translation():
    """MDS recovers a configuration only up to a rigid motion, so distances --
    not coordinates -- are what must match."""
    rng = np.random.default_rng(53)
    X = rng.standard_normal((12, 2))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    a = _pdist(np.asarray(mds(X, k=2)["coords"]))
    b = _pdist(np.asarray(mds(X @ R.T + 5.0, k=2)["coords"]))
    assert a == pytest.approx(b, abs=1e-8)


def test_mdspl_accepts_a_distance_matrix_directly():
    """A square symmetric input is treated as D, not re-measured as coordinates."""
    X = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    D = _pdist(X)
    from_coords = _pdist(np.asarray(mds(X, k=2)["coords"]))
    from_dist = _pdist(np.asarray(mds(D, k=2)["coords"]))
    assert from_coords == pytest.approx(from_dist, abs=1e-8)
    # 3-4-5 triangle: the hypotenuse must come back as 5.
    assert from_dist[1, 2] == pytest.approx(5.0, abs=1e-8)


def test_mdspl_one_dimensional_data_needs_only_one_dimension():
    """Collinear points have a single non-zero eigenvalue."""
    X = np.array([[0.0], [1.0], [2.0], [5.0]])
    ev = np.asarray(mds(X, k=2)["eigenvalues"])
    assert ev[0] > 1e-6
    assert abs(ev[1]) < 1e-8


def test_mdspl_stress_falls_as_dimensions_are_added():
    """More dimensions can only fit the distances better."""
    rng = np.random.default_rng(59)
    X = rng.standard_normal((15, 4))
    s = [mds(X, k=k)["stress"] for k in (1, 2, 3, 4)]
    assert all(s[i] >= s[i + 1] - 1e-12 for i in range(3))


def test_mdspl_reports_shape():
    rng = np.random.default_rng(61)
    r = mds(rng.standard_normal((9, 3)), k=2)
    assert r["n"] == 9 and r["k"] == 2
    assert np.asarray(r["coords"]).shape == (9, 2)
