"""ESL chapter 14 against R: prcomp, factanal (tight), KL NMF properties and arm parity."""

import math

from morie.fn import esl_nmf, esl_pca_svd, esl_pca_transform, factor_analysis_ml


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def X14():
    return [
        [
            math.sin(t) + 0.3 * math.cos(3 * t),
            math.cos(2 * t) + 0.5 * math.sin(t),
            math.sin(3 * t) - 0.2 * math.sin(t),
            math.cos(t) + 0.4 * math.sin(2 * t),
            0.6 * math.sin(t) + 0.2 * math.cos(5 * t),
        ]
        for t in range(1, 41)
    ]


def test_pca_and_factor_analysis():
    X = X14()
    m = esl_pca_svd(X, 3)
    for a, b in zip(m["eigenvalues"], (0.995343760720, 0.610326740774, 0.558286723727)):  # prcomp sdev^2
        assert close(a, b, 1e-11)
    assert all(close(a, b) for a, b in zip(esl_pca_transform(m, X), m["scores"]))
    f = factor_analysis_ml(X, n_factors=1, tol=1e-14, max_iter=100000)
    mu = [sum(r[j] for r in X) / 40 for j in range(5)]
    S = [sum((r[j] - mu[j]) ** 2 for r in X) / 39 for j in range(5)]
    u = [1 - float(c) / s for c, s in zip(f.communalities.tolist(), S)]
    ref = (0.0420442604889737, 0.8218942654366636, 0.9621242415868327, 0.9993448528544326, 0.1346083649223915)
    assert all(close(a, b, 1e-8) for a, b in zip(u, ref))  # factanal(factr = 1, pgtol = 0)


def test_kl_nmf():
    X = [[abs(math.sin(i * j / 3)) * 3 + (i + j) % 3 for j in range(1, 7)] for i in range(1, 9)]
    r = esl_nmf(X, 2, loss="kl", max_iter=5000, tol=1e-14)
    path = r["divergence_path"]
    assert all(b <= a + 1e-12 for a, b in zip(path, path[1:]))  # Lee-Seung monotonicity
    W = [r["W"][i * 2 : (i + 1) * 2] for i in range(8)]
    H = [r["H"][k * 6 : (k + 1) * 6] for k in range(2)]
    WH = [[sum(W[i][k] * H[k][j] for k in range(2)) for j in range(6)] for i in range(8)]
    for k in range(2):
        g = sum(W[i][k] * (X[i][j] / WH[i][j] - 1) for i in range(8) for j in range(6))
        assert abs(g) < 1e-4  # stationarity in H (summed over columns)
    for i in range(8):
        for k in range(2):
            gw = sum(H[k][j] * (X[i][j] / WH[i][j] - 1) for j in range(6))
            assert abs(gw) < 1e-4  # stationarity in W: sum_j h_kj (x_ij / (WH)_ij - 1) = 0
    tot = sum(map(sum, X))
    assert abs(sum(map(sum, WH)) - tot) < 1e-8 * tot  # KL updates keep sum WH = sum X at the fixed point
    const = sum(X[i][j] - (X[i][j] * math.log(X[i][j]) if X[i][j] > 0 else 0) for i in range(8) for j in range(6))
    assert close(r["loglik"] + const, -r["kl_divergence"], 1e-10)  # eq 14.73 is -D up to a constant
    fro = esl_nmf(X, 2)
    assert fro["loss"] == "frobenius" and "kl_divergence" not in fro
