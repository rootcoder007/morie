"""BFGS, L-BFGS-B and Nelder-Mead: exact optima, KKT conditions and R optim references."""

from morie.fn.bfgsmin import bfgs_minimize
from morie.fn.lbfgsb import lbfgsb_minimize
from morie.fn.neldmd import nelder_mead


def rb(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rbg(x):
    return [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)]


def ext(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def extg(x):
    g = [0.0] * len(x)
    for i in range(len(x) - 1):
        g[i] += -400 * x[i] * (x[i + 1] - x[i] ** 2) - 2 * (1 - x[i])
        g[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
    return g


X10 = [-1.2 if i % 2 == 0 else 1.0 for i in range(10)]


def test_bfgs_reaches_the_rosenbrock_minimum():
    for g in (rbg, None):
        r = bfgs_minimize(rb, [-1.2, 1.0], g)
        assert r["converged"] and max(abs(v - 1) for v in r["x"]) < 1e-9 and r["fun"] < 1e-18
    r = bfgs_minimize(ext, X10, extg)
    assert r["converged"] and max(abs(v) for v in extg(r["x"])) <= 1e-8 and max(abs(v - 1) for v in r["x"]) < 1e-8
    assert r["n_iter"] == 77  # R arm BfgsMinimize: 77


def test_lbfgsb_box_solutions_match_optim_and_kkt():
    r = lbfgsb_minimize(rb, [-1.2, 1.0], rbg, [-2, -2], [0.5, 2])
    assert abs(r["x"][0] - 0.5) < 1e-12 and abs(r["x"][1] - 0.25) < 1e-9 and abs(r["fun"] - 0.25) < 1e-12
    assert (r["n_iter"], r["n_fev"]) == (24, 54)  # R arm LbfgsbMinimize: 24, 54
    r = lbfgsb_minimize(ext, X10, extg, [-2] * 10, [0.8] * 10)
    # optim(method = "L-BFGS-B"): value 6.001016, par 0.8 0.6658853 0.4603305 0.2244283 0.06099439 ...
    ref = [
        0.8,
        0.6658853,
        0.4603305,
        0.2244283,
        0.06099439,
        0.01386156,
        0.01029636,
        0.01020575,
        0.01000428,
        0.000100357,
    ]
    assert abs(r["fun"] - 6.001016) < 1e-6 and max(abs(a - b) for a, b in zip(r["x"], ref)) < 5e-6
    g = extg(r["x"])  # KKT: the first coordinate sits on its upper bound with g < 0; the rest are stationary
    assert r["x"][0] == 0.8 and g[0] < 0 and max(abs(v) for v in g[1:]) < 1e-4
    assert (r["n_iter"], r["n_fev"]) == (34, 73)
    r = lbfgsb_minimize(ext, X10, extg)
    assert max(abs(v - 1) for v in r["x"]) < 1e-5 and r["message"].startswith("relative reduction")


def test_lbfgsb_projects_the_start_and_handles_one_sided_bounds():
    r = lbfgsb_minimize(
        lambda x: (x[0] - 3) ** 2 + (x[1] + 1) ** 2, [10.0, 10.0], None, [-float("inf"), 0.0], [2.0, float("inf")]
    )
    assert abs(r["x"][0] - 2) < 1e-12 and abs(r["x"][1]) < 1e-12 and r["converged"]


def test_nelder_mead_minimisers():
    r = nelder_mead(rb, [-1.2, 1.0])
    assert r["converged"] and max(abs(v - 1) for v in r["x"]) < 1e-8
    # the first 40 iterations are a fixed path (termination counts sit at the ftol noise floor)
    r = nelder_mead(rb, [-1.2, 1.0], max_iter=40)
    assert (
        r["n_fev"] == 78
        and max(abs(a - b) for a, b in zip(r["x"], [-0.18575362041593402, 0.011429483070955149])) < 1e-12
    )
    q = nelder_mead(
        lambda x: (x[0] - 1) ** 2 + 2 * (x[1] + 2) ** 2 + 3 * (x[2] - 0.5) ** 2 + x[0] * x[1], [0.0, 0.0, 0.0]
    )
    # stationary point of the quadratic: x1 = 16/7, x2 = -18/7, x3 = 1/2
    assert max(abs(a - b) for a, b in zip(q["x"], [16 / 7, -18 / 7, 0.5])) < 1e-7
    q40 = nelder_mead(
        lambda x: (x[0] - 1) ** 2 + 2 * (x[1] + 2) ** 2 + 3 * (x[2] - 0.5) ** 2 + x[0] * x[1],
        [0.0, 0.0, 0.0],
        max_iter=40,
    )
    assert (
        q40["n_fev"] == 72
        and max(abs(a - b) for a, b in zip(q40["x"], [2.2759718688762636, -2.589707882818465, 0.5123693758516892]))
        < 1e-12
    )
    assert abs(q["fun"] - ((16 / 7 - 1) ** 2 + 2 * (-18 / 7 + 2) ** 2 - 16 / 7 * 18 / 7)) < 1e-12
    assert [round(v, 6) for v in nelder_mead(lambda x: (x[0] - 1) ** 2 + (x[1] + 2) ** 2, [0.0, 0.0])["x"]] == [
        1.0,
        -2.0,
    ]


def _model_path_min(x, g, lo, hi, B):
    # first local minimiser of m(t) = g (x(t) - x) + (x(t) - x) B (x(t) - x) / 2 along x(t) = P(x - t g):
    # m is quadratic between breakpoints, so minimise it exactly segment by segment
    n = len(x)
    inf = float("inf")
    tb = [((x[i] - hi[i]) / g[i] if g[i] < 0 else (x[i] - lo[i]) / g[i]) if g[i] != 0 else inf for i in range(n)]
    pts = sorted({0.0} | {t for t in tb if 0 < t < inf}) + [inf]
    for a, b in zip(pts, pts[1:]):
        za = [min(max(x[i] - a * g[i], lo[i]), hi[i]) - x[i] for i in range(n)]
        d = [-g[i] if tb[i] > a else 0.0 for i in range(n)]
        Bd = [sum(B[i][j] * d[j] for j in range(n)) for i in range(n)]
        f1 = sum(gi * di for gi, di in zip(g, d)) + sum(zi * bi for zi, bi in zip(za, Bd))
        f2 = sum(di * bi for di, bi in zip(d, Bd))
        if f1 >= 0:
            return [x[i] + za[i] for i in range(n)]
        if f2 > 0 and -f1 / f2 < b - a:
            return [x[i] + za[i] - f1 / f2 * d[i] for i in range(n)]
    return [min(max(x[i] - pts[-2] * g[i], lo[i]), hi[i]) for i in range(n)]


def test_generalized_cauchy_point_is_the_first_path_minimiser():
    from morie.fn.lbfgsb import _cauchy, _compact

    S = [[0.3, -0.2, 0.1, 0.4], [-0.1, 0.5, 0.2, -0.3], [0.2, 0.1, -0.4, 0.1]]
    Y = [[1.1, -0.3, 0.4, 0.9], [-0.2, 1.4, 0.3, -0.8], [0.5, 0.2, -1.2, 0.3]]
    theta = 1.7
    W, M = _compact(S, Y, theta)
    B = [
        [theta * (i == j) - sum(W[i][a] * M[a][b] * W[j][b] for a in range(6) for b in range(6)) for j in range(4)]
        for i in range(4)
    ]
    x = [0.2, -0.4, 0.9, 0.1]
    lo, hi = [-0.3, -1.0, 0.0, -0.2], [0.5, 0.3, 1.0, 0.6]
    for g in ([2.0, -1.5, 3.0, 0.7], [-0.4, 2.5, -0.3, 1.9], [0.9, 0.2, 2.2, -2.8]):
        xc, _ = _cauchy(x, g, lo, hi, theta, W, M)
        ref = _model_path_min(x, g, lo, hi, B)
        assert max(abs(a - b) for a, b in zip(xc, ref)) < 1e-12
