"""Gauss-Newton nonlinear least squares against R nls and lm (Hedderich, Sec. 3.7.12 examples)."""

from morie.fn.nlsgn import nonlinear_least_squares


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_quadratic_book_example():
    # book p. 142: 189/35 - 92/35 x + 4/7 x^2, RSS 3.829; linear in the parameters, so lm gives the SEs
    r = nonlinear_least_squares(
        lambda x, t: t[0] + t[1] * x + t[2] * x * x, [1, 2, 3, 4, 5], [4, 1, 3, 5, 6], [1, 1, 1]
    )
    assert r["converged"]
    for a, b in zip(r["coefficients"], (189 / 35, -92 / 35, 4 / 7)):
        assert close(a, b, 1e-9)
    assert close(r["rss"], 134 / 35, 1e-12)
    for a, b in zip(r["se"], (2.96744238119533366, 2.26138841224155529, 0.36977654587270725)):
        assert close(a, b, 1e-7)


def test_exponential_book_example():
    # book p. 147: nls(y ~ a * b^x) gives 1.602022, 1.998596, RSS 1.225082
    r = nonlinear_least_squares(lambda x, t: t[0] * t[1] ** x, [1, 2, 3, 4, 5], [3, 7, 12, 26, 51], [1, 1])
    assert r["converged"]
    # nls(..., control = nls.control(tol = 1e-9)): 1.602023650300048, 1.998595286780294, RSS 1.2250822723304109
    assert close(r["coefficients"][0], 1.602023650300048008, 1e-6) and close(
        r["coefficients"][1], 1.998595286780293634, 1e-7
    )
    assert r["rss"] <= 1.225082272330410893 * (1 + 1e-13)
    assert close(r["se"][0], 0.126191623123091456, 1e-5) and close(r["se"][1], 0.033383451913671848, 1e-5)


def test_step_halving_from_a_poor_start():
    # from (1, 0.5) the full Gauss-Newton step overshoots; halving still reaches the nls optimum
    r = nonlinear_least_squares(lambda x, t: t[0] * t[1] ** x, [1, 2, 3, 4, 5], [3, 7, 12, 26, 51], [1, 0.5])
    assert r["converged"] and r["rss"] <= 1.225082272330410893 * (1 + 1e-13)
    assert close(r["coefficients"][0], 1.602023650300048008, 1e-6)


def test_converges_at_the_rounding_floor():
    # tol = 0 never passes the relative-offset test, as on arm64 where the
    # Gauss-Newton decrease stays one ulp above zero; the fit must converge
    r = nonlinear_least_squares(lambda x, t: t[0] * t[1] ** x, [1, 2, 3, 4, 5], [3, 7, 12, 26, 51], [1, 0.5], tol=0)
    assert r["converged"] and r["rss"] <= 1.225082272330410893 * (1 + 1e-13)
