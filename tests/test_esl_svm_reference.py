"""SVM family against libsvm (e1071): linear and kernel classifiers, epsilon-regression; kernel logistic regression."""

import math

from morie.fn import esl_svc, esl_svm_kernel
from morie.fn.eslplg import esl_penalized_logistic
from morie.fn.eslsvr import esl_svr


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def data():
    X = [[math.sin(t), math.cos(3 * t) + (t % 3) * 0.5] for t in range(1, 41)]
    return X, [1 if math.sin(2 * t) + r[0] > 0.2 else -1 for r, t in zip(X, range(1, 41))]


def test_classifiers_reach_the_optimum():
    X, y = data()
    r = esl_svc(X, y, C=1.0, tol=1e-10)
    w = [float(v) for v in r["w"]]
    for a, b in zip(w + [float(r["b"])], (1.213774815839871, 0.123255850258580, -0.218576046711221)):
        assert close(a, b, 1e-6)  # e1071::svm(kernel = "linear", tolerance = 1e-10)
    obj = 0.5 * sum(v * v for v in w) + sum(
        max(0.0, 1 - t * (w[0] * x[0] + w[1] * x[1] + float(r["b"]))) for x, t in zip(X, y)
    )
    assert obj < 28.0982623 + 1e-6  # the simplified SMO it replaces stopped at 28.1233
    k = esl_svm_kernel(X, y, C=1.0, kernel="rbf", gamma=0.7, tol=1e-10)
    assert close(float(k["b"]), -0.398046272259155, 1e-6) and k["n_support"] == 32


def test_svr_equals_libsvm():
    X = [[math.sin(t), math.cos(3 * t)] for t in range(1, 41)]
    y = [2 * r[0] - r[1] + 0.3 * math.cos(7 * t) for r, t in zip(X, range(1, 41))]
    r = esl_svr(X, y, C=1.0, epsilon=0.2, kernel="rbf", gamma=0.7, tol=1e-10, newdata=[[0.2, 0.5], [-0.5, 0.9]])
    for a, b in zip([r["b"]] + r["prediction"], (0.0802678275040895, -0.2376537861308012, -1.8255432832230458)):
        assert close(a, b, 1e-6)
    assert r["n_support"] == 20
    assert all(abs(c) < 1e-9 for c, yi, fi in zip(r["coef"], y, r["fitted"]) if abs(yi - fi) < 0.2 - 1e-6)
    lin = esl_svr(X, y, C=1.0, epsilon=0.2, kernel="linear", tol=1e-10, newdata=[[0.2, 0.5]])
    assert close(lin["b"], -0.0225538837328316, 1e-6) and close(lin["prediction"][0], -0.1253864791905822, 1e-6)


def test_kernel_logistic_regression_is_penalised_logistic_on_the_kernel_basis():
    # ESL eq 12.32: f(x) = b0 + sum alpha_i K(x, x_i) with penalty lambda/2 alpha' K alpha
    X, y = data()
    yb = [1 if v > 0 else 0 for v in y]
    K = [[math.exp(-0.7 * sum((a - b) ** 2 for a, b in zip(u, v))) for v in X] for u in X]
    N = [[1.0] + row for row in K]
    Om = [[0.0] * 41] + [[0.0] + row for row in K]
    r = esl_penalized_logistic(N, yb, Om, 0.5)
    assert r["converged"]
    # stationarity: N'(y - p) = lambda Omega theta, to the ~1e-8 rounding floor of the near-singular RBF Gram matrix
    res = [a - b for a, b in zip(yb, r["prob"])]
    for c in range(41):
        lhs = sum(N[i][c] * res[i] for i in range(40))
        rhs = 0.5 * sum(Om[c][d] * r["theta"][d] for d in range(41))
        assert abs(lhs - rhs) < 1e-6
