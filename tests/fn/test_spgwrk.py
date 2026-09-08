"""spgwrk -- GWR kernel weight functions.

Sources: Charlton, GWR White Paper pp. 6-7 (gaussian, bisquare, adaptive);
spgwr R/gwr.gauss.R, R/gwr.bisquare.R, R/tricube.R; GWmodel R/gw.weight.r
(all four, incl. the boxcar); Schabenberger & Gotway Sec. 5.3.2 pp. 240-241.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.spgwrk import schabenberger_gwr_kernels as kern

H = 2.0
D = np.array([0.0, 0.5, 1.0, 1.5, 1.9999, 2.0, 2.5, 4.0])


def test_gaussian_matches_the_printed_formula():
    w = kern(D, H, "gaussian")["weights"]
    assert np.allclose(w, np.exp(-0.5 * (D / H) ** 2))


def test_gaussian_is_one_at_the_regression_point():
    assert kern(0.0, H, "gaussian")["weights"] == pytest.approx(1.0)


def test_gaussian_never_truncates():
    """The white paper's Gaussian has no finite support; only the other three do.

    The weight decays but never reaches zero by any rule of the kernel. Far
    enough out it underflows to 0.0 in double precision -- that is the float
    format running out, not a support boundary, so the range checked here
    stops short of underflow and the boxcar comparison makes the difference
    explicit.
    """
    w = kern(np.array([0.0, H, 5 * H, 20 * H]), H, "gaussian")["weights"]
    assert np.all(w > 0)
    assert kern(D, H, "gaussian")["truncated"] is False
    # at the same distances a truncated kernel is already identically zero
    assert kern(np.array([5 * H, 20 * H]), H, "bisquare")["weights"].tolist() == [0.0, 0.0]


def test_gaussian_underflows_rather_than_truncating():
    """Distinguishes the float limit from a support boundary."""
    assert kern(40.0 * H, H, "gaussian")["weights"] == 0.0
    assert kern(np.array([40.0 * H]), H, "gaussian")["truncated"] is False


def test_bisquare_matches_the_printed_formula():
    w = kern(D, H, "bisquare")["weights"]
    want = np.where(D < H, (1 - (D / H) ** 2) ** 2, 0.0)
    assert np.allclose(w, want)


def test_tricube_matches_the_printed_formula():
    w = kern(D, H, "tricube")["weights"]
    want = np.where(D < H, (1 - (D / H) ** 3) ** 3, 0.0)
    assert np.allclose(w, want)


def test_boxcar_is_an_indicator():
    w = kern(D, H, "boxcar")["weights"]
    assert np.allclose(w, (D < H).astype(float))
    assert set(np.unique(w)) <= {0.0, 1.0}


def test_bisquare_and_tricube_are_not_the_same_kernel():
    """Guards against one silently being implemented as the other."""
    b = kern(D, H, "bisquare")["weights"]
    t = kern(D, H, "tricube")["weights"]
    assert not np.allclose(b, t)
    # tricube is flatter near the centre and drops faster near the edge
    assert t[1] > b[1]


def test_truncated_kernels_vanish_at_and_beyond_the_bandwidth():
    for k in ("bisquare", "tricube", "boxcar"):
        w = kern(np.array([H, H + 1e-9, 2 * H]), H, k)["weights"]
        assert np.all(w == 0.0), k
        assert kern(D, H, k)["truncated"] is True


def test_bisquare_and_tricube_reach_zero_smoothly_but_boxcar_does_not():
    """The white paper's "near-Gaussian" property: a vanishing edge derivative."""
    eps = 1e-6
    assert kern(H - eps, H, "bisquare")["weights"] < 1e-11
    assert kern(H - eps, H, "tricube")["weights"] < 1e-16
    assert kern(H - eps, H, "boxcar")["weights"] == 1.0


def test_weights_decrease_with_distance():
    """Sec. 6.1.3.1 p. 317 states this as the one requirement on the kernel."""
    d = np.linspace(0, 1.9, 25)
    for k in ("gaussian", "bisquare", "tricube"):
        w = kern(d, H, k)["weights"]
        assert np.all(np.diff(w) <= 1e-15), k


def test_density_form_is_the_same_kernel_rescaled():
    """Sec. 5.3.2 prints the Gaussian as a density; the constant is immaterial."""
    plain = kern(D, H, "gaussian")["weights"]
    dens = kern(D, H, "gaussian", normalized=True)["weights"]
    assert np.allclose(dens, plain / (H * np.sqrt(2 * np.pi)))
    ratio = dens / plain
    assert np.allclose(ratio, ratio[0])


def test_density_form_rejected_for_truncated_kernels():
    for k in ("bisquare", "tricube", "boxcar"):
        with pytest.raises(ValueError):
            kern(D, H, k, normalized=True)


def test_adaptive_bandwidth_admits_exactly_k_neighbours():
    rs = np.random.RandomState(11)
    d = np.abs(rs.uniform(0, 20, 50))
    d[0] = 0.0
    for k in (3, 10, 25):
        r = kern(d, k, "bisquare", adaptive=True)
        assert r["n_nonzero"] == k
        assert r["bandwidth"] > 0


def test_adaptive_reports_the_distance_it_actually_used():
    d = np.array([0.0, 1.0, 2.0, 3.0, 9.0])
    r = kern(d, 3, "boxcar", adaptive=True)
    assert r["bandwidth"] == pytest.approx(2.0, rel=1e-6)
    assert r["n_nonzero"] == 3
    assert r["adaptive"] is True


def test_adaptive_and_fixed_differ_on_an_uneven_sample():
    """The whole point of an adaptive bandwidth is that it is not fixed."""
    d = np.array([0.0, 0.1, 0.2, 8.0, 9.0, 10.0])
    fixed = kern(d, 1.0, "bisquare")["n_nonzero"]
    adapt = kern(d, 5, "bisquare", adaptive=True)["n_nonzero"]
    assert fixed == 3
    assert adapt == 5


def test_n_nonzero_counts_what_it_says():
    r = kern(D, H, "bisquare")
    assert r["n_nonzero"] == int(np.sum(np.asarray(r["weights"]) > 0))


@pytest.mark.parametrize("bad", ["epanechnikov", "quartic", "triangular", ""])
def test_unknown_kernel_rejected(bad):
    with pytest.raises(ValueError):
        kern(D, H, bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_bad_bandwidth_rejected(bad):
    with pytest.raises(ValueError):
        kern(D, bad, "gaussian")


def test_negative_distance_rejected():
    with pytest.raises(ValueError):
        kern(np.array([1.0, -0.5]), H, "gaussian")


@pytest.mark.parametrize("bad", [0, -2, 99])
def test_bad_neighbour_count_rejected(bad):
    with pytest.raises(ValueError):
        kern(np.arange(5.0), bad, "bisquare", adaptive=True)


def test_payload_shape_matches_input():
    for shape in [(8,), (4, 2)]:
        d = np.abs(np.random.RandomState(2).standard_normal(shape))
        assert np.shape(kern(d, H, "tricube")["weights"]) == shape
