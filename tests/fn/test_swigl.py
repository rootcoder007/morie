"""swigl: SwiGLU gated activation (Shazeer 2020).

    SwiGLU(x) = SiLU(xW + b) * (xV + c)
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.swigl import swiglu_activation as sg


def _silu(z):
    return z / (1.0 + np.exp(-z))


def test_swigl_matches_the_closed_form():
    rng = np.random.default_rng(2301)
    x = rng.standard_normal((4, 3))
    W = rng.standard_normal((3, 5))
    V = rng.standard_normal((3, 5))
    b = rng.standard_normal(5)
    c = rng.standard_normal(5)
    got = np.asarray(sg(x, W=W, V=V, b=b, c=c)["tensor"])
    assert got == pytest.approx(_silu(x @ W + b) * (x @ V + c))


def test_swigl_default_projections_are_the_identity():
    """With no W/V it reduces to elementwise SiLU(x) * x."""
    x = np.array([[1.0, -2.0, 0.0, 3.0]])
    assert np.asarray(sg(x)["tensor"]) == pytest.approx(_silu(x) * x)


def test_swigl_reports_the_two_halves_separately():
    rng = np.random.default_rng(2311)
    x = rng.standard_normal((3, 4))
    r = sg(x)
    assert np.asarray(r["tensor"]) == pytest.approx(
        np.asarray(r["gate"]) * np.asarray(r["up"])
    )


def test_swigl_zero_up_projection_kills_the_signal():
    """The gated half of a gated linear unit: a zero up-projection gives
    exactly zero whatever the gate says."""
    rng = np.random.default_rng(2317)
    x = rng.standard_normal((3, 4))
    V = np.zeros((4, 4))
    W = np.eye(4)
    assert np.asarray(sg(x, W=W, V=V)["tensor"]) == pytest.approx(np.zeros((3, 4)))


def test_swigl_is_linear_in_the_up_projection():
    """Only the gate half is nonlinear, so doubling V doubles the output."""
    rng = np.random.default_rng(2321)
    x = rng.standard_normal((3, 4))
    W = rng.standard_normal((4, 4))
    V = rng.standard_normal((4, 4))
    assert np.asarray(sg(x, W=W, V=2 * V)["tensor"]) == pytest.approx(
        2 * np.asarray(sg(x, W=W, V=V)["tensor"])
    )


def test_swigl_is_not_linear_in_the_gate():
    """If it were, SwiGLU would collapse to a bilinear form and the
    activation would contribute nothing."""
    rng = np.random.default_rng(2333)
    x = rng.standard_normal((3, 4))
    W = rng.standard_normal((4, 4))
    V = rng.standard_normal((4, 4))
    assert not np.allclose(
        np.asarray(sg(x, W=2 * W, V=V)["tensor"]),
        2 * np.asarray(sg(x, W=W, V=V)["tensor"]),
    )


def test_swigl_silu_properties_show_through():
    """SiLU(0) = 0; SiLU is negative for small negative inputs, unlike ReLU;
    and it approaches the identity for large positive ones."""
    x = np.array([[0.0, -1.0, 30.0]])
    gate = np.asarray(sg(x)["gate"])
    assert gate[0, 0] == pytest.approx(0.0)
    assert gate[0, 1] < 0.0
    assert gate[0, 2] == pytest.approx(30.0, rel=1e-9)


def test_swigl_rejects_only_one_projection():
    """W without V is a caller error, not a case to guess at."""
    with pytest.raises(ValueError, match="both W and V or neither"):
        sg(np.zeros((2, 3)), W=np.eye(3))
