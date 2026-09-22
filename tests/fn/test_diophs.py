"""Tests for diophs.diophantine."""

from morie.fn import _array_core as np

from morie.fn.diophs import diophantine


def test_diophs_basic():
    """Test basic functionality."""
    a, b, c = 6, 9, 27  # gcd(6,9)=3 divides 27; 6*(-2) + 9*5 = -12+45 = 33 (no); 6*3+9*1=27 -> x=3,y=1
    # Extended Euclid on (6,9): g=3, x0=-1, y0=1 since -1*6+1*9=3 -> scale by 9 -> x0=-9, y0=9? Let's compute by hand below.
    # We will instead pick a known solvable example and verify by independent arithmetic.
    a, b, c = 4, 6, 10  # gcd=2 divides 10. 4*4 + 6*(-1) = 16-6 = 10 -> (x=4,y=-1) is one solution.
    result = diophantine(a, b, c)
    assert isinstance(result, dict)
    assert "estimate" in result
    assert "solvable" in result
    assert result["solvable"] is True or result["solvable"] == True
    # Independent verification: gcd(4,6)=2; 2 divides 10 -> solvable.
    g_indep = 2
    assert int(result["gcd"]) == g_indep
    # Independent verification: a*x + b*y must equal c.
    xv = int(np.asarray(result["x"]).item())
    yv = int(np.asarray(result["y"]).item())
    assert a * xv + b * yv == c
    # Step sizes are b/g and -a/g for integer-parametric family.
    xs_step = int(np.asarray(result["x_step"]).item())
    ys_step = int(np.asarray(result["y_step"]).item())
    assert xs_step == b // g_indep
    assert ys_step == -(a // g_indep)


def test_diophs_edge():
    """Test edge cases."""
    # Case 1: no solution because gcd does not divide c.
    a, b, c = 4, 6, 7  # gcd(4,6)=2 does not divide 7 -> not solvable
    result = diophantine(a, b, c)
    assert isinstance(result, dict)
    assert result["solvable"] is False or result["solvable"] == False
    assert float(result["estimate"]) == 0.0
    assert result["x"] is None
    assert result["y"] is None
    # Case 2: solvable example with a negative coefficient.
    a, b, c = -4, 10, 6  # gcd(4,10)=2 divides 6 -> solvable.
    # Independent solution check via brute force search.
    found = None
    for t in range(-50, 51):
        for s in range(-50, 51):
            if a * t + b * s == c:
                found = (t, s)
                break
        if found is not None:
            break
    assert found is not None  # sanity: there is a solution
    result = diophantine(a, b, c)
    assert result["solvable"] is True or result["solvable"] == True
    assert float(result["estimate"]) == 1.0
    xv = int(np.asarray(result["x"]).item())
    yv = int(np.asarray(result["y"]).item())
    assert a * xv + b * yv == c
    # Parametric family: x = x0 + t*(b/g), y = y0 - t*(a/g). Verify two
    # members of the family both satisfy the equation.
    g_indep = 2
    for t in (0, 1, -2, 7):
        assert a * (xv + t * (b // g_indep)) + b * (yv - t * (a // g_indep)) == c
