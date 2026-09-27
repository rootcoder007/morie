"""ESL chapter 17 against R: glasso (with zero constraints), the Poisson log-linear Ising fit, norm::em.norm."""

import math

from morie.fn.esleggm import esl_ggm_fit
from morie.fn.eslglso import esl_graphical_lasso
from morie.fn.eslisg import esl_ising_fit
from morie.fn.eslmem import esl_mvn_em_missing
from morie.fn.eslprr import esl_precision_regression


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_modified_regression_book_example():
    S = [[10, 1, 5, 4], [1, 10, 2, 6], [5, 2, 10, 3], [4, 6, 3, 10]]
    g = esl_ggm_fit(S, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
    # ESL p. 634: Sigma_13 = 1.31, Sigma_24 = 0.87; glasso(S, rho = 0, zero = (1,3),(2,4)) to 1e-15
    assert close(g["Sigma"][0][2], 1.3142061401609193) and close(g["Sigma"][1][3], 0.8704715678852768)
    th = g["Theta"]
    for (a, b), v in {
        (0, 0): 0.1196574473628500418,
        (0, 1): -0.0078589574550913203,
        (0, 3): -0.047178879043352287,
        (2, 3): -0.032374932123883496,
        (3, 3): 0.128584031254505959,
    }.items():
        assert close(th[a][b], v)
    assert th[0][2] == 0 and th[1][3] == 0
    pc = esl_precision_regression(th)
    assert pc["partial_correlation"][0][2] == 0 and close(pc["residual_variance"][0], 1 / th[0][0])
    assert close(pc["partial_correlation"][0][1], 0.0078589574550913203 / math.sqrt(0.1196574473628500418 * 0.104770135356931499))


def test_graphical_lasso_equals_glasso():
    i = range(1, 61)
    Z = [
        [math.sin(t), math.cos(2 * t) + 0.5 * math.sin(t), math.sin(3 * t), math.cos(t) - 0.3 * math.cos(2 * t)]
        for t in i
    ]
    mu = [sum(r[j] for r in Z) / 60 for j in range(4)]
    S = [[sum((r[a] - mu[a]) * (r[b] - mu[b]) for r in Z) / 60 for b in range(4)] for a in range(4)]
    th = esl_graphical_lasso(S, 0.1)["Theta"]
    ref = {
        (0, 0): 1.74385207753239735,
        (0, 1): -0.32897601072540467,
        (1, 1): 1.49541713002190790,
        (1, 3): 0.13499180745482239,
        (2, 2): 1.578917840269016,
        (3, 3): 1.53868376697536280,
    }
    for (a, b), v in ref.items():
        assert close(th[a][b], v, 1e-10)
    assert th[0][2] == 0 and th[2][3] == 0


def test_ising_equals_poisson_loglinear():
    Xb = [
        [
            int(math.sin(t) > 0),
            int(math.sin(t) + math.cos(3 * t) > 0.2),
            int(math.cos(2 * t) > 0),
            int(math.sin(5 * t) + math.sin(t) > 0),
        ]
        for t in range(1, 201)
    ]
    f = esl_ising_fit(Xb, [(0, 1), (1, 2), (2, 3), (0, 3)])
    ref = (
        -2.615028710276642521,
        -2.436619044465905048,
        -0.065027669973812929,
        -1.613271775976331313,
        3.025619922017412922,
        0.241095368508791597,
        -0.094166225899906641,
        3.247145874856569581,
    )
    got = f["main"] + [v for _, v in f["edges"]]
    assert all(close(a, b, 1e-10) for a, b in zip(got, ref))
    assert f["converged"] and f["iterations"] <= 8  # Newton with the exact information converges quadratically


def test_em_equals_norm():
    i = list(range(1, 61))
    Z = [[math.sin(t), math.cos(2 * t) + 0.5 * math.sin(t), math.sin(3 * t)] for t in i]
    X = [[r[0], None if t % 5 == 0 else r[1], None if t % 7 == 0 else r[2]] for r, t in zip(Z, i)]
    e = esl_mvn_em_missing(X)
    for a, b in zip(e["mean"], (0.0272421447271645328, 0.0267546087048901107, -0.0095027235347008073)):
        assert close(a, b, 1e-10)
    assert close(e["cov"][0][1], 0.20974805481014477, 1e-10) and close(e["cov"][1][2], 0.1188797433131135312, 1e-10)
