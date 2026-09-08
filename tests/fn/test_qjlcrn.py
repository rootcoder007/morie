"""qjlcrn: Johnson-Lindenstrauss random projection.

The generated test imported `qjl_compression`, a name the module never
exported. Rewritten against johnson_lindenstrauss and anchored on the
lemma itself: the projection preserves norms in expectation.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.qjlcrn import johnson_lindenstrauss, target_dimension


def test_embedding_has_the_requested_shape():
    A = [[float(i + j) for j in range(8)] for i in range(5)]
    r = johnson_lindenstrauss(A, 3, seed=1)
    E = np.asarray(r["embedding"])
    assert len(E) == 5
    assert len(np.asarray(E[0])) == 3
    assert r["k"] == 3
    assert r["d"] == 8


def test_norms_are_preserved_on_average_over_many_projections():
    """E||Ax||^2 = ||x||^2 is the scaling the estimator exists to provide.
    Averaging over seeds must approach the true norm; a single projection
    need not.
    """
    x = [[1.0, -2.0, 3.0, 0.5, -1.5, 2.5]]
    true = sum(v * v for v in x[0])
    est = []
    for s in range(60):
        E = np.asarray(johnson_lindenstrauss(x, 4, seed=s)["embedding"])
        est.append(sum(float(v) ** 2 for v in np.asarray(E[0])))
    assert sum(est) / len(est) == pytest.approx(true, rel=0.15)


def test_target_dimension_grows_as_epsilon_shrinks():
    """k = O(log n / eps^2): a tighter distortion needs more dimensions."""
    assert target_dimension(1000, 0.1)["k"] > target_dimension(1000, 0.5)["k"]
