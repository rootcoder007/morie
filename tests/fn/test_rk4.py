"""Tests for morie.fn.rk4."""
import pytest

from morie.fn import _array_core as np

from morie.fn.rk4 import rk4


def test_rk4_smoke():
    result = rk4(f=lambda t, y: -0.5 * y, y0=np.array([1.0]), t_span=(0.0, 5.0))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.rk4 import cheatsheet

    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0


def test_rk4_is_the_stability_polynomial_on_a_linear_ode():
    """On y' = lam y one RK4 step multiplies y by
    R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24 with z = h lam, exactly;
    after N steps y_N = R(z)^N, and a 2-D system decouples."""
    lam, N = -0.5, 40
    r = rk4(f=lambda t, y: lam * y, y0=np.array([1.0, 3.0]), t_span=(0.0, 5.0), n_steps=N)
    z = 5.0 / N * lam
    R = 1 + z + z * z / 2 + z ** 3 / 6 + z ** 4 / 24
    fs = [float(v) for v in r.extra["final_state"]]
    assert fs == pytest.approx([R ** N, 3 * R ** N], rel=1e-13)
    assert r.value == pytest.approx(R ** N, rel=1e-13)
    # and R(z)^N sits within the O(h^4) global error of exp(-2.5)
    import math
    assert abs(R ** N - math.exp(-2.5)) < 1e-6
