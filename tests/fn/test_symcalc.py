import math

import pytest

from morie.fn.symcalc import (
    _ev,
    _parse,
    matrix_symbolic,
    ode_symbolic,
    risch_integration,
    symbolic_limit,
)


def _f(s, **env):
    return _ev(_parse(s), env)


def _deriv(s, x, v, h=1e-5, **env):
    """Central difference of the expression s in the variable x."""
    return (_f(s, **{x: v + h}, **env) - _f(s, **{x: v - h}, **env)) / (2 * h)


def test_risch_integrals_differentiate_back_to_the_integrand():
    for q in (
        "(x^2 + 1)/(x^3 - x)",
        "x/(x^2 + 2*x + 1)",
        "1/(x^2 + 1)",
        "x^3/(x^2 - 1)",
        "1/(x^2*(x + 1))",
        "1/(x^2 + 1)^2",
        "(x^4 + 1)/(x^3 + x)",
    ):
        r = risch_integration(q)
        assert r.rational and r.out_of_scope is None and r.verified
        for v in (1.63, 2.71, 3.4):  # above every branch point, so the logarithms are real
            assert _deriv(r.integral, "x", v) == pytest.approx(_f(q, x=v), rel=1e-6)


def test_risch_parts_add_up_and_match_the_known_forms():
    r = risch_integration("1/(x^2 + 1)")
    assert (r.polynomial_part, r.rational_part, r.log_part) == ("0", "0", "atan(x)")
    r = risch_integration("x/(x^2 + 2*x + 1)")
    # 1/(x + 1) - 1/(x + 1)^2 integrates to log(x + 1) + 1/(x + 1)
    assert r.log_part == "log(x + 1)"
    for v in (0.4, 2.2):
        assert _f(r.rational_part, x=v) == pytest.approx(-v / (v + 1), abs=1e-12)
    r = risch_integration("x^3/(x^2 - 1)")
    for v in (1.3, 2.5):
        assert _f(r.polynomial_part, x=v) == pytest.approx(0.5 * v * v, abs=1e-12)
        assert _f(r.integral, x=v) == pytest.approx(
            0.5 * v * v + 0.5 * math.log(abs(v + 1)) + 0.5 * math.log(abs(v - 1)), abs=1e-9
        )


def test_risch_reports_what_is_out_of_scope():
    r = risch_integration("exp(x^2)")
    assert r.integral is None and not r.rational and r.generators == ["exp"]
    assert "transcendental tower" in r.out_of_scope
    r = risch_integration("x^(1/2)")
    assert r.generators == ["algebraic"] and "algebraic extensions" in r.out_of_scope
    assert risch_integration("sin(x)/x").out_of_scope.startswith("the transcendental tower")


def test_limits_agree_with_the_values_of_the_function_nearby():
    cases = [
        ("sin(x)/x", 0.0, "both", 1.0, "series"),
        ("(1 - cos(x))/x^2", 0.0, "both", 0.5, "series"),
        ("(exp(x) - 1)/x", 0.0, "both", 1.0, "series"),
        ("tan(x)/x", 0.0, "both", 1.0, "series"),
        ("(x^3 - 1)/(x - 1)", 1.0, "both", 3.0, "series"),
        ("(x^2 + 3*x)/(2*x^2 - 1)", "inf", "both", 0.5, "degrees"),
        ("x*exp(-x)", "inf", "both", 0.0, "order"),
        ("log(x)/x", "inf", "both", 0.0, "order"),
        ("sin(x)/x", "inf", "both", 0.0, "squeeze"),
    ]
    for e, x0, side, want, method in cases:
        r = symbolic_limit(e, "x", x0, side=side)
        assert r.method == method
        assert r.limit == pytest.approx(want, abs=1e-12)
        v = 1e6 if x0 == "inf" else float(x0) + 1e-5
        assert _f(e, x=v) == pytest.approx(want, abs=2e-4)


def test_infinite_and_one_sided_limits():
    assert symbolic_limit("1/x", "x", 0, side="+").limit == math.inf
    assert symbolic_limit("1/x", "x", 0, side="-").limit == -math.inf
    with pytest.raises(ValueError):
        symbolic_limit("1/x", "x", 0)
    assert symbolic_limit("exp(x)/x^5", "x", "inf").limit == math.inf
    assert symbolic_limit("(3*x^3 - x)/(x^2 + 1)", "x", "-inf").limit == -math.inf
    assert symbolic_limit("x^2*exp(-x^2)", "x", "inf").limit == 0.0
    assert symbolic_limit("3*x + 1", "x", 2).limit == 7.0
    assert symbolic_limit("3*x + 1", "x", 2).method == "substitution"
    with pytest.raises(ValueError):
        symbolic_limit("(1 + 1/x)^x", "x", "inf")  # 1^inf: none of the rules decides it
    with pytest.raises(ValueError):
        symbolic_limit("sin(x)", "x", "inf")
    with pytest.raises(ValueError):
        symbolic_limit("x", "x", 0, side="up")


def _det(A):
    n = len(A)
    if n == 1:
        return A[0][0]
    return math.fsum(
        (-1.0) ** j * A[0][j] * _det([[A[i][k] for k in range(n) if k != j] for i in range(1, n)]) for j in range(n)
    )


def test_symbolic_determinant_inverse_and_char_poly():
    r = matrix_symbolic([["a", "b"], ["c", "d"]])
    env = {"a": 2.0, "b": 1.0, "c": 1.0, "d": 3.0}  # a positive discriminant, so the eigenvalues are real
    A = [[env["a"], env["b"]], [env["c"], env["d"]]]
    assert _f(r.determinant, **env) == pytest.approx(_det(A), abs=1e-12)
    inv = [_f(s, **env) for s in r.inverse]
    for i in range(2):
        for j in range(2):
            got = math.fsum(A[i][k] * inv[2 * k + j] for k in range(2))
            assert got == pytest.approx(1.0 if i == j else 0.0, abs=1e-12)
    for t in (0.3, -1.7):
        shifted = [[A[i][j] - (t if i == j else 0.0) for j in range(2)] for i in range(2)]
        assert _f(r.char_poly, t=t, **env) == pytest.approx(_det(shifted), abs=1e-12)
    assert r.trace == "a + d"
    for lam in r.eigenvalues:
        v = _f(lam, **env)
        assert _f(r.char_poly, t=v, **env) == pytest.approx(0.0, abs=1e-9)


def test_numeric_eigenvalues_and_larger_matrices():
    r = matrix_symbolic([[2, 1], [1, 2]])
    assert r.eigenvalues_numeric == [1.0, 3.0]
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
    r = matrix_symbolic(A)
    assert _f(r.determinant) == pytest.approx(_det(A), abs=1e-12)
    assert len(r.eigenvalues_numeric) == 3
    for lam in r.eigenvalues_numeric:
        shifted = [[A[i][j] - (lam if i == j else 0.0) for j in range(3)] for i in range(3)]
        assert _det(shifted) == pytest.approx(0.0, abs=1e-9)
    assert r.eigenvalues is None  # degree 3, no closed form is offered
    with pytest.raises(ValueError):
        matrix_symbolic([[1, 2, 3], [4, 5, 6]])


def _ode_holds(sol, rhs, xs, c=1.0):
    for v in xs:
        yv = _f(sol, x=v, C=c)
        dy = (_f(sol, x=v + 1e-6, C=c) - _f(sol, x=v - 1e-6, C=c)) / 2e-6
        if not (math.isfinite(yv) and math.isfinite(dy)):
            return False
        if abs(dy - _f(rhs, x=v, y=yv)) > 1e-5 * max(1.0, abs(dy)):
            return False
    return True


def test_ode_solutions_satisfy_the_equation():
    for rhs, classes, xs, c in (
        ("2*x*y", ["separable", "linear"], (0.8, 1.4, 2.3), 0.7),
        ("y + x", ["linear"], (0.8, 1.4, 2.3), 0.7),
        ("2*y/x + x^2", ["linear"], (0.8, 1.4, 2.3), 0.7),
        ("x*y + x*y^3", ["separable", "bernoulli"], (0.3, 0.6, 0.9), 300.0),
        ("(x + y)/x", ["linear", "homogeneous"], (0.8, 1.4, 2.3), 0.7),
        ("exp(x)*y", ["separable", "linear"], (0.8, 1.4, 2.3), 0.7),
    ):
        r = ode_symbolic(rhs)
        assert r.classes == classes
        assert r.form == "explicit"
        assert _ode_holds(r.solution, rhs, xs, c=c)


def test_ode_classification_criteria_and_implicit_solutions():
    r = ode_symbolic("y' = 2*x*y")
    assert r.solution == "C*exp(x^2)" and r.coefficients["p"] == "2*x" and r.coefficients["q"] == "0"
    b = ode_symbolic("x*y + x*y^3")
    assert b.exponent == 3.0
    s = ode_symbolic("y^2*x")
    assert s.form == "implicit" and s.classes == ["separable"]
    # the implicit solution -1/y = C + x^2/2 differentiates to y' = x y^2
    for v, c in ((0.9, 0.5), (1.7, -0.3)):
        yv = -1.0 / (c + 0.5 * v * v)
        assert _f(s.solution.split("=")[0], y=yv) == pytest.approx(_f(s.solution.split("=")[1], x=v, C=c), abs=1e-9)
    e = ode_symbolic(M="2*x*y", N="x^2 + 1")
    assert e.classes == ["exact"] and e.solution == "x^2*y + y = C"
    assert ode_symbolic(M="y", N="x^2").classes == []
    with pytest.raises(ValueError):
        ode_symbolic(M="y")
    with pytest.raises(ValueError):
        ode_symbolic("z' = x")
