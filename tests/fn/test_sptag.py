"""sptag: pairwise agreement matrix.

Armstrong et al., section 3.2.2, "90th US Senate Agreement Scores", printed
p.88 -- verified against the PDF. The module previously cited "Armstrong
Ch 8"; that book has six chapters.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.sptag import spatial_agreement as sa


def test_sptag_canonical_example_from_the_module():
    """The example recorded in the module's own CANONICAL TEST comment."""
    M = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    A = np.asarray(sa(M)["agreement"])
    assert A[0, 1] == 1.0          # rows 0 and 1 are identical
    assert A[0, 2] == 0.0          # row 2 is the exact complement
    assert A[2, 2] == 1.0          # self-agreement


def test_sptag_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(23)
    A = np.asarray(sa(rng.integers(0, 2, (8, 20)).astype(float))["agreement"])
    assert np.allclose(A, A.T)
    assert np.allclose(np.diag(A), 1.0)


def test_sptag_agreement_is_a_proportion():
    """Two members agreeing on 3 of 4 votes score exactly 0.75."""
    M = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
    assert np.asarray(sa(M)["agreement"])[0, 1] == pytest.approx(0.75)


def test_sptag_absences_are_excluded_from_the_denominator():
    """Only votes where BOTH members were present count.

    Here the pair overlaps on 2 votes and agrees on both, so agreement is
    1.0 -- not 2/4, which is what counting absences as disagreements gives.
    """
    M = np.array([[1.0, 0.0, np.nan, np.nan], [1.0, 0.0, 1.0, 0.0]])
    assert np.asarray(sa(M)["agreement"])[0, 1] == pytest.approx(1.0)


def test_sptag_no_shared_votes_is_nan_not_zero():
    """Never voting together means agreement is undefined, not 'never agreed'."""
    M = np.array([[1.0, 0.0, np.nan, np.nan], [np.nan, np.nan, 1.0, 0.0]])
    assert np.isnan(np.asarray(sa(M)["agreement"])[0, 1])


def test_sptag_mean_agreement_is_the_off_diagonal_mean():
    M = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    r = sa(M)
    A = np.asarray(r["agreement"])
    iu = np.triu_indices(3, k=1)
    assert r["mean_agreement"] == pytest.approx(float(np.nanmean(A[iu])))


def test_sptag_reports_shape():
    r = sa(np.zeros((5, 7)))
    assert r["n"] == 5 and r["m"] == 7
