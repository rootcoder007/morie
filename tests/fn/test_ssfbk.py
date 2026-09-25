"""Tests for morie.fn.ssfbk."""

import pytest

from morie.fn import _array_core as np

from morie.fn.ssfbk import ssfbk


def test_ssfbk_smoke():
    A = np.array([[0, 1], [-2, -3]], dtype=float)
    B = np.array([[0], [1]], dtype=float)
    poles = np.array([-1, -2])
    result = ssfbk(A=A, B=B, poles=poles)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.ssfbk import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0


def test_ssfbk_companion_form_gain():
    """In controllable canonical form K is the desired minus the open-loop
    characteristic coefficients: (s^2 + 2s + 5)(s + 4) = s^3 + 6s^2 + 13s
    + 20 against s^3 + 1.5s^2 + 3s + 2 gives K = (18, 10, 4.5); scipy's
    place_poles returns the same gain."""
    A = [[0, 1, 0], [0, 0, 1], [-2.0, -3.0, -1.5]]
    r = ssfbk(A, [0, 0, 1.0], [-1 + 2j, -1 - 2j, -4])
    for got, want in zip(r.extra["K"], [18.0, 10.0, 4.5]):
        assert float(got) == pytest.approx(want, abs=1e-12)
    eig = sorted(r.extra["closed_loop_eigenvalues"].tolist(), key=lambda z: (z.real, z.imag))
    for got, want in zip(eig, [-4, -1 - 2j, -1 + 2j]):
        assert abs(got - want) < 1e-10


def test_ssfbk_general_matches_place_poles():
    """A non-canonical system against scipy.signal.place_poles."""
    A = [[1.0, 2, 0.5], [0.3, -1, 2], [0, 1.5, 0.2]]
    r = ssfbk(A, [1.0, 0.2, -0.7], [-2, -3, -0.5])
    ref = [-10.923049350003915, -23.526343016234012, -30.469025647501017]
    for got, want in zip(r.extra["K"], ref):
        assert float(got) == pytest.approx(want, rel=1e-12)


def test_ssfbk_rejects_multi_input():
    with pytest.raises(ValueError, match="single input"):
        ssfbk([[1.0, 0], [0, 2.0]], [[1.0, 0], [0, 1.0]], [-1, -2])
