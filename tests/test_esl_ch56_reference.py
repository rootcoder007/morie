"""ESL chapters 5-6 against R: Reinsch smoothing spline, weighted lm, mclust, e1071, multinom, optim, lm."""

import math

from morie.fn import (
    esl_em_gmm,
    esl_local_linear,
    esl_mallows_cp,
    esl_nadaraya_watson,
    esl_naive_bayes,
    esl_natural_spline,
    esl_smoothing_spline,
    esl_svm_kernel,
    kernel_ridge_regression,
)
from morie.fn.eslfwe import esl_fwer
from morie.fn.esllgl import esl_local_logistic
from morie.fn.eslplg import esl_penalized_logistic
from morie.fn.eslrbf import esl_rbf_network
from morie.fn.eslvcm import esl_varying_coef


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def xy():
    x = sorted(((7 * t) % 23) / 23 + t / 100 for t in range(1, 26))
    return x, [math.sin(12 * (v + 0.2)) / (v + 0.2) + 0.3 * math.cos(5 * t) for t, v in enumerate(x, 1)]


def test_smoothers():
    x, y = xy()
    s = esl_smoothing_spline(x, y, 0.001)  # (I + lambda Q R^-1 Q')^-1 y, Reinsch
    for a, b in zip(
        s["estimate"][:4], (-2.8655887051872546, -2.0690428290326972, -1.6258749972063486, -1.1970995169258036)
    ):
        assert close(a, b, 1e-11)
    assert close(s["effective_df"], 5.3426937102204342, 1e-11) and close(s["rss"], 4.4518194766928030, 1e-11)
    ll = esl_local_linear([0.3, 0.6], x, y, 0.2)  # lm(y ~ I(x - x0), weights = epanechnikov)
    assert close(ll["values"][0], -0.63596327831420862) and close(ll["slopes"][1], -10.525653987328347228)
    nw = esl_nadaraya_watson([0.3, 0.6], x, y, 0.2)
    assert close(nw["values"][0], -0.77842873317765782) and close(nw["values"][1], 0.006458624327503458)
    k = kernel_ridge_regression([[v] for v in x], y, lam=0.1, bandwidth=0.2, x_eval=[[0.3], [0.6]])
    assert close(k["y_hat"][0], -0.565622851739059795) and close(k["y_hat"][1], -0.026796905212836009)


def test_natural_spline_and_cp():
    b = esl_natural_spline([0.1, 0.35, 0.6, 0.9], [0.2, 0.5, 0.8])["basis"]
    # N3 = d1 - d2 with d_k = ((x - xi_k)_+^3 - (x - xi_K)_+^3)/(xi_K - xi_k)
    assert close(b[5], 0.15**3 / 0.6) and close(b[11], (0.7**3 - 0.1**3) / 0.6 - (0.4**3 - 0.1**3) / 0.3)
    c = esl_mallows_cp(10.0, 3, 50, 0.5)
    assert close(c["estimate"], 0.26) and close(c["cp_classical"], -24.0)


def test_mixture_bayes_svm():
    z = [math.sin(t) * 0.4 + (3 if t % 2 else 0) for t in range(1, 41)]
    m = esl_em_gmm([[v] for v in z], k=2, reg=0.0)  # mclust::Mclust(z, G = 2, modelName = "V")
    mus = sorted(float(v[0]) for v in m["mu"])
    assert close(mus[0], 0.018154417810861530, 1e-9) and close(mus[1], 3.019809810341028466, 1e-9)
    sig = sorted(float(v[0][0]) for v in m["sigma"])
    assert close(sig[0], 0.080981470300304767, 1e-7) and close(sig[1], 0.081793629847064758, 1e-7)
    X = [[math.sin(t), math.cos(3 * t) + (t % 3)] for t in range(1, 31)]
    nb = esl_naive_bayes(X, [t % 3 for t in range(1, 31)], [[0.2, 1.0]])  # Gaussian, MLE variances
    for a, b in zip(nb["log_posterior"], (-3.4270370202619498, -1.9641204550416957, -3.6230859821947980)):
        assert close(a, b, 1e-8)
    Xs = [[math.sin(t), math.cos(3 * t) + (t % 3) * 0.5] for t in range(1, 41)]
    ys = [1 if math.sin(2 * t) + r[0] > 0.2 else -1 for r, t in zip(Xs, range(1, 41))]
    s = esl_svm_kernel(Xs, ys, C=1.0, kernel="rbf", gamma=0.7, tol=1e-8, max_passes=500)
    assert close(s["b"], -0.398046272259155, 1e-6)  # e1071::svm rho, tolerance = 1e-10


def test_new_ch56_methods():
    i = list(range(1, 41))
    x = [math.sin(t) + 0.1 * t / 10 for t in i]
    yb = [1 if math.sin(3 * t) + x[t - 1] > 0.3 else 0 for t in i]
    N = [[1.0, v, v * v, v**3] for v in x]
    Om = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 4.0, 6.0], [0, 0, 6.0, 12.0]]
    r = esl_penalized_logistic(N, yb, Om, 0.5)  # optim(BFGS) on the penalised log-likelihood
    for a, b in zip(
        r["theta"], (-1.295262125853519075, 4.219150558200408518, -0.602041164046019261, 0.095061923090123832)
    ):
        assert close(a, b, 1e-7)
    assert close(r["penalized_loglik"], -11.872385602248254344, 1e-10)
    X = [[math.cos(2 * t), math.sin(5 * t)] for t in i]
    z = [((7 * t) % 40) / 40 for t in i]
    y = [1 + z[k] * X[k][0] - (1 - z[k]) * X[k][1] + 0.1 * math.cos(9 * t) for k, t in enumerate(i)]
    v = esl_varying_coef(X, z, y, [0.3, 0.7], 0.35)  # lm(y ~ X, weights = epanechnikov(|z - z0| / 0.35))
    for a, b in zip(
        v["coefficients"][0] + v["coefficients"][1],
        (
            1.01369601567464462,
            0.35736764439982477,
            -0.69206275946687890,
            0.98308304405220848,
            0.72381010867275386,
            -0.31086813300694366,
        ),
    ):
        assert close(a, b)
    XX = [[math.sin(t), math.cos(3 * t)] for t in i]
    g = [(t * 7) % 3 for t in i]
    lg = esl_local_logistic(XX, g, [[0.2, 0.3], [-0.5, 0.8]], 1.0)  # weighted nnet::multinom
    for a, b in zip(
        lg["prob"][0] + lg["prob"][1],
        (
            0.30580338856933753,
            0.38233137071712425,
            0.31186524071353816,
            0.21298316785666599,
            0.37583969978620452,
            0.41117713235712949,
        ),
    ):
        assert close(a, b, 1e-7)
    yr = [math.sin(2 * t) for t in i]
    cen, sc = [[0, 0], [1, 1], [-1, 0.5]], [0.8, 1.0, 1.2]
    b1 = esl_rbf_network(XX, yr, cen, sc, query=[[0.2, 0.3]])  # lm on the basis
    for a, b in zip(
        b1["coefficients"], (0.20342086160818848, -0.21045409303914872, -0.83284203448504524, 0.11965929479195532)
    ):
        assert close(a, b)
    b2 = esl_rbf_network(XX, yr, cen, sc, normalized=True, query=[[0.2, 0.3]])
    for a, b in zip(b2["coefficients"], (0.453143574359092305, -0.655609262139223659, -0.059510247707946336)):
        assert close(a, b)
    assert close(b1["fitted"][0], -0.194568199295301142) and close(b2["fitted"][0], 0.091361122278410797)
    f = esl_fwer(0.05, 10)
    assert close(f["fwer_independent"], 1 - 0.95**10) and close(f["per_test_sidak"], 1 - 0.95**0.1)


def test_householder_refuses_collinear_design():
    import pytest

    from morie.fn.linsys import _householder_ls

    # the third column is 0.1 c1 + 0.3 c2: Householder leaves rounding residue, not an exact zero,
    # and back-substituting on it would return huge garbage coefficients
    with pytest.raises(ValueError, match="rank deficient"):
        _householder_ls([[1, t, 0.1 + 0.3 * t] for t in (1.0, 2.0, 3.0, 5.0, 7.5)], [1, 2, 3, 4, 5])
