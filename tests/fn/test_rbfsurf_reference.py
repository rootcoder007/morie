"""rbfsurf: interpolation, polynomial reproduction, Rippa identity and multiscale behaviour."""

import math

import pytest

from morie.fn.rbfsurf import radial_basis, rbf_grid, rbf_interpolate, rbf_loocv, rbf_multiscale

X = [(0.1, 0.2), (0.9, 0.1), (0.5, 0.6), (0.2, 0.9), (0.8, 0.8), (0.4, 0.3), (0.6, 0.35)]
Y = [math.sin(3 * a) + b * b for a, b in X]


def test_kernels():
    assert radial_basis([0.0, 2.0], "thin_plate") == [0.0, 4 * math.log(2)]
    assert radial_basis([2.0], "polyharmonic", order=4) == [-(16 * math.log(2))]
    assert radial_basis([2.0], "polyharmonic", order=5) == [-32.0]
    assert radial_basis([1.0], "multiquadric", epsilon=2.0) == [-math.sqrt(5)]
    with pytest.raises(ValueError):
        radial_basis([1.0], "bogus")


def test_interpolation_and_polynomial_reproduction():
    for kern, deg, kw in (
        ("thin_plate", 1, {}),
        ("cubic", 1, {}),
        ("quintic", 2, {}),
        ("gaussian", -1, {"epsilon": 2.0}),
        ("multiquadric", 0, {"epsilon": 1.5}),
        ("wendland", -1, {"support": 1.5}),
        ("polyharmonic", 1, {"order": 3}),
    ):
        r = rbf_interpolate(X, Y, X, kernel=kern, degree=deg, **kw)
        assert r.prediction == pytest.approx(Y, abs=1e-8)
    lin = [2 + 3 * a - b for a, b in X]
    r = rbf_interpolate(X, lin, [(0.3, 0.7), (1.2, -0.4)], kernel="thin_plate", degree=1)
    assert r.prediction == pytest.approx([2 + 0.9 - 0.7, 2 + 3.6 + 0.4], abs=1e-9)
    sm = rbf_interpolate(X, Y, X, kernel="thin_plate", smoothing=0.5)
    assert max(abs(a - b) for a, b in zip(sm.prediction, Y)) > 1e-3
    iso = rbf_interpolate(X, Y, [(0.5, 0.5)], kernel="gaussian", degree=-1, epsilon=1.0)
    scaled = rbf_interpolate(X, Y, [(0.5, 0.5)], kernel="gaussian", degree=-1, epsilon=1.0, transform=[[1, 0], [0, 1]])
    assert iso.prediction == pytest.approx(scaled.prediction)


def test_rippa_equals_refitting():
    r = rbf_loocv(X, Y, [1.0, 3.0], kernel="gaussian")
    for e, rm in zip([1.0, 3.0], r.rmse):
        errs = []
        for k in range(len(X)):
            Xo = X[:k] + X[k + 1 :]
            Yo = Y[:k] + Y[k + 1 :]
            p = rbf_interpolate(Xo, Yo, [X[k]], kernel="gaussian", epsilon=e, degree=-1).prediction[0]
            errs.append(Y[k] - p)
        assert rm == pytest.approx(math.sqrt(sum(v * v for v in errs) / len(errs)), rel=1e-8)


def test_multiscale_and_grid():
    m = rbf_multiscale(X, Y, X, [2.0, 1.0, 0.5])
    assert m.prediction == pytest.approx(Y, abs=1e-8) and m.residual_rms[-1] < 1e-8
    g = rbf_grid(X, Y, [0.1, 0.5], [0.2, 0.6], kernel="thin_plate")
    assert g.surface[0][0] == pytest.approx(Y[0], abs=1e-9) and g.surface[1][1] == pytest.approx(Y[2], abs=1e-9)
