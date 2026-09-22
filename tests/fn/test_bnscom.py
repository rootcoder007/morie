"""Tests for bnscom.bound_compliance."""

from morie.fn import _array_core as np

from morie.fn.bnscom import bound_compliance


def test_bnscom_basic():
    """Test basic functionality with a hand-constructed IV dataset."""
    # y, D, Z are binary-constrained per the docstring: D and Z must be coded 0/1.
    # y is continuous; design a tiny dataset and compute the expected quantities
    # directly from the formulas stated in the docstring.

    # Rows: (y, D, Z)
    data = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [3.0, 1.0, 1.0],
        [4.0, 1.0, 1.0],
        [5.0, 0.0, 1.0],
        [6.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [3.5, 0.0, 0.0],
    ])

    y = data[:, 0]
    D = data[:, 1]
    Z = data[:, 2]

    result = bound_compliance(y, D, Z)

    # n
    n = 10

    # First-stage shares
    n_z1 = 6.0
    n_z0 = 4.0
    pd1 = 4.0 / n_z1            # P(D=1 | Z=1)
    pd0 = 1.0 / n_z0            # P(D=1 | Z=0)
    pi_c = pd1 - pd0            # complier share
    pi_a = pd0                  # always-taker share
    pi_n = 1.0 - pd1            # never-taker share

    # E[y D | Z=z] and E[y (1-D) | Z=z]
    # Z=1: D=1 for rows 0..3 (y sum = 10), D=0 for rows 4,5 (y sum = 11)
    e_yD_z1 = 10.0 / n_z1
    e_y1mD_z1 = 11.0 / n_z1
    # Z=0: D=1 for row 6 (y=0), D=0 for rows 7..9 (y sum = 7.5)
    e_yD_z0 = 0.0 / n_z0
    e_y1mD_z0 = 7.5 / n_z0

    e1c = (e_yD_z1 - e_yD_z0) / pi_c
    e0c = (e_y1mD_z0 - e_y1mD_z1) / pi_c
    late = e1c - e0c

    y0 = 0.0   # min(y)
    y1 = 6.0   # max(y)

    rest = 1.0 - pi_c
    lo = pi_c * late + rest * (y0 - y1)
    hi = pi_c * late + rest * (y1 - y0)

    # The result is a dict-like RichResult; check documented keys/values.
    assert isinstance(result, dict)

    for key in ("lower", "upper", "width", "estimate", "late",
                "pi_c", "pi_a", "pi_n", "e1c", "e0c", "n"):
        assert key in result

    assert result["n"] == n
    assert result["pi_c"] == pi_c
    assert result["pi_a"] == pi_a
    assert result["pi_n"] == pi_n
    assert result["e1c"] == e1c
    assert result["e0c"] == e0c
    assert result["late"] == late
    assert result["lower"] == lo
    assert result["upper"] == hi
    assert result["width"] == hi - lo
    assert result["estimate"] == 0.5 * (lo + hi)


def test_bnscom_edge():
    """Test edge cases: minimal valid dataset still of correct arity."""
    y = np.array([1.0, 2.0])
    D = np.array([1.0, 0.0])
    Z = np.array([1.0, 0.0])

    result = bound_compliance(y, D, Z)

    assert isinstance(result, dict)
    # Both instrument values present, so no "only one value" error.
    assert result["n"] == 2
    # First stage is positive, so monotonicity check passes.
    assert result["pi_c"] > 0.0
