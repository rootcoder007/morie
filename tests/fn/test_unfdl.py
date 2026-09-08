"""unfdl: metric multidimensional unfolding.

Schoenemann, P. H. (1970). "On metric multidimensional unfolding."
*Psychometrika*, 35(3), 349-366 -- in the library, verified from the PDF
("PSYCHOMETRIKA VOL. 35, NO. 3 ... ON METRIC MULTIDIMENSIONAL UNFOLDING ...
THE OHIO STATE UNIVERSITY"). Schoenemann states the problem as "locating two
sets of points in a joint space, given the distances between them", which is
exactly what the tests below plant and recover.

Armstrong et al. cover unfolding in Ch 4 (rating-scale data, printed p.107)
and Ch 5 (binary choice, printed p.129); the module formerly cited
"Armstrong Ch 7", which does not exist -- that book has six chapters.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.unfdl import unfolding_analysis as unfold


def _cross(X, Y):
    """Row-by-column Euclidean distances ||x_i - y_j||."""
    d = X[:, None, :] - Y[None, :, :]
    return np.sqrt((d**2).sum(-1))


def test_unfdl_recovers_a_planted_joint_configuration():
    """Plant X and Y, hand it only the cross-distances, require them back.

    Unfolding recovers a joint space only up to a rigid motion, so the thing
    that must match is the DISTANCE matrix, not the coordinates.
    """
    rng = np.random.default_rng(1103)
    X = rng.standard_normal((12, 2))
    Y = rng.standard_normal((5, 2))
    delta = _cross(X, Y)
    # n_iter is raised past the default: the default 100 stops on the
    # iteration cap rather than on tol, leaving ~4e-04 of error. See the
    # convergence test below, which pins that behaviour explicitly.
    r = unfold(delta, k=2, n_iter=500)
    got = _cross(np.asarray(r["X"]), np.asarray(r["Y"]))
    assert got == pytest.approx(delta, abs=1e-5)


def test_unfdl_stress_is_zero_on_an_exactly_embeddable_configuration():
    rng = np.random.default_rng(1109)
    delta = _cross(rng.standard_normal((10, 2)), rng.standard_normal((4, 2)))
    assert unfold(delta, k=2, n_iter=500)["stress"] == pytest.approx(0.0, abs=1e-5)


def test_unfdl_default_now_converges_across_many_configurations():
    """Regression guard on the raised n_iter default.

    The docstring claimed a "closed-form Schoenemann solution"; the function
    actually iterates with a tol break, and the old default of n_iter=100 left
    24 of these 30 configurations unconverged -- worst case 1.25e-01 of error
    in recovered cross-distance where convergence gives ~1e-05. The default is
    now 1000. If someone lowers it again, this fails.
    """
    unconverged = 0
    for seed in range(1100, 1130):
        rng = np.random.default_rng(seed)
        delta = _cross(rng.standard_normal((10, 2)), rng.standard_normal((4, 2)))
        d = unfold(delta, k=2)                       # default n_iter
        ref = unfold(delta, k=2, n_iter=6000)
        e_def = np.abs(_cross(np.asarray(d["X"]), np.asarray(d["Y"])) - delta).max()
        e_ref = np.abs(_cross(np.asarray(ref["X"]), np.asarray(ref["Y"])) - delta).max()
        if e_def > 10 * max(e_ref, 1e-12):
            unconverged += 1
    assert unconverged == 0, f"{unconverged}/30 unconverged at the default n_iter"


def test_unfdl_reports_the_shape_it_was_given():
    rng = np.random.default_rng(1117)
    r = unfold(_cross(rng.standard_normal((9, 2)), rng.standard_normal((6, 2))), k=2)
    assert r["n_resp"] == 9 and r["n_stim"] == 6 and r["k"] == 2
    assert np.asarray(r["X"]).shape == (9, 2)
    assert np.asarray(r["Y"]).shape == (6, 2)


def test_unfdl_is_invariant_to_a_rigid_motion_of_the_truth():
    """Rotating and translating the planted configuration cannot change the
    recovered distances -- the joint space is identified only up to that."""
    rng = np.random.default_rng(1123)
    X = rng.standard_normal((11, 2))
    Y = rng.standard_normal((4, 2))
    th = 0.9
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    a = unfold(_cross(X, Y), k=2)
    b = unfold(_cross(X @ R.T + 3.0, Y @ R.T + 3.0), k=2)
    da = _cross(np.asarray(a["X"]), np.asarray(a["Y"]))
    db = _cross(np.asarray(b["X"]), np.asarray(b["Y"]))
    assert da == pytest.approx(db, abs=1e-6)


def test_unfdl_one_dimensional_preferences_need_one_dimension():
    """Coombs's original case: respondents and stimuli on a single scale."""
    X = np.array([[-2.0], [0.0], [1.5], [3.0]])
    Y = np.array([[-1.0], [0.5], [2.5]])
    r = unfold(_cross(X, Y), k=1)
    got = _cross(np.asarray(r["X"]), np.asarray(r["Y"]))
    assert got == pytest.approx(_cross(X, Y), abs=1e-6)


def test_unfdl_more_dimensions_cannot_fit_worse():
    rng = np.random.default_rng(1129)
    delta = _cross(rng.standard_normal((14, 3)), rng.standard_normal((6, 3)))
    s = [unfold(delta, k=k)["stress"] for k in (1, 2, 3)]
    assert all(s[i] >= s[i + 1] - 1e-9 for i in range(len(s) - 1))


def test_unfdl_preserves_the_preference_ordering_of_each_respondent():
    """The substantive claim of unfolding: a respondent's ideal point is
    nearer the stimuli they prefer. Ranking stimuli by recovered distance
    must reproduce the ranking in the input dissimilarities.
    """
    rng = np.random.default_rng(1151)
    X = rng.standard_normal((8, 2))
    Y = rng.standard_normal((5, 2))
    delta = _cross(X, Y)
    r = unfold(delta, k=2)
    got = _cross(np.asarray(r["X"]), np.asarray(r["Y"]))
    for i in range(delta.shape[0]):
        assert np.argsort(got[i]).tolist() == np.argsort(delta[i]).tolist()
