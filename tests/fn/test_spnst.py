"""spnst -- Hughes-Oliver point-source model, Schabenberger & Gotway Sec. 8.2.1."""

from morie.fn import _array_core as np
import pytest

from morie.fn.spnst import schabenberger_nonstationary_cov as nst


def _sites(n=40, seed=7):
    return np.random.RandomState(seed).uniform(0, 10, (n, 2))


SRC = np.array([5.0, 5.0])


def test_reduces_to_exponential_when_theta2_theta3_zero():
    s = _sites()
    r = nst(s, source=SRC, theta1=0.4)
    assert np.allclose(r["correlation"], np.exp(-0.4 * r["separation"]))


def test_practical_range_is_three_over_theta1():
    r = nst(_sites(), source=SRC, theta1=0.4)
    assert r["practical_range"] == pytest.approx(7.5)
    assert np.exp(-0.4 * r["practical_range"]) == pytest.approx(np.exp(-3))


def test_nonstationarity_same_separation_different_correlation():
    near = np.array([[5.0, 5.0], [5.0, 6.0]])
    far = np.array([[5.0, 12.0], [5.0, 13.0]])
    cn = nst(near, source=SRC, theta1=0.4, theta2=0.3, theta3=0.2)
    cf = nst(far, source=SRC, theta1=0.4, theta2=0.3, theta3=0.2)
    assert cn["separation"][0, 1] == pytest.approx(cf["separation"][0, 1])
    assert cn["correlation"][0, 1] != cf["correlation"][0, 1]


def test_equidistant_pair_matches_the_printed_range_formula():
    """p. 423: c_i = c_j = c gives alpha = 3 exp(-theta3 c)/theta1."""
    d0 = 2.0
    pair = np.array([[5.0 + d0, 5.0], [5.0, 5.0 + d0]])
    r = nst(pair, source=SRC, theta1=0.4, theta3=0.2)
    alpha = 3.0 * np.exp(-0.2 * d0) / 0.4
    h = r["separation"][0, 1]
    assert r["correlation"][0, 1] == pytest.approx(np.exp(-3 * h / alpha))


def test_correlation_diagonal_is_one_and_matrix_symmetric():
    r = nst(_sites(), source=SRC, theta1=0.3, theta2=0.1, theta3=0.1)
    assert np.allclose(np.diag(r["correlation"]), 1.0)
    assert np.allclose(r["correlation"], r["correlation"].T)


def test_sill_scales_correlation_into_covariance():
    s = _sites()
    r = nst(s, source=SRC, theta1=0.3, sill=4.0)
    assert np.allclose(r["nonstationary_cov"], 4.0 * r["correlation"])


def test_psd_constraints_are_necessary_but_not_sufficient():
    """The book's warning made concrete: a conforming set that is not PSD."""
    s = np.random.RandomState(3).uniform(0, 10, (20, 2))
    bad = nst(s, source=np.array([2.0, 8.0]), theta1=0.021, theta2=0.457,
              theta3=0.037)
    assert bad["min_eigenvalue"] < 0
    assert bad["valid"] is False
    assert "warning" in bad


def test_valid_case_reports_no_warning():
    r = nst(_sites(), source=SRC, theta1=0.4, theta2=0.05, theta3=0.02)
    assert r["valid"] is True
    assert "warning" not in r


def test_default_source_is_the_centroid():
    s = _sites()
    r = nst(s, theta1=0.4)
    assert np.allclose(r["source"], s.mean(axis=0))


def test_anisotropy_changes_the_answer():
    s = _sites()
    A = np.diag([1.0, 3.0])
    iso = nst(s, source=SRC, theta1=0.3, theta2=0.1, theta3=0.1)
    ani = nst(s, source=SRC, theta1=0.3, theta2=0.1, theta3=0.1, anisotropy=A)
    assert not np.allclose(iso["correlation"], ani["correlation"])


@pytest.mark.parametrize("kw", [
    {"theta1": 0.0}, {"theta1": -0.4}, {"theta1": 0.4, "theta2": -0.1},
    {"theta1": 0.4, "theta3": -0.1}])
def test_rejects_out_of_range_parameters(kw):
    with pytest.raises(ValueError):
        nst(_sites(), source=SRC, **kw)


def test_rejects_bad_sill_and_source_dimension():
    with pytest.raises(ValueError):
        nst(_sites(), source=SRC, theta1=0.4, sill=0.0)
    with pytest.raises(ValueError):
        nst(_sites(), source=np.array([1.0, 2.0, 3.0]), theta1=0.4)
