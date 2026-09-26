"""morie.diagnostics against R's influence.measures, car::vif and lmtest."""

from morie import diagnostics as D
from morie.fn import _array_core as np

X1 = [2.1, 2.9, 2.0, 3.3, 2.6, 3.0, 2.4, 2.7, 2.5, 3.1, 1.9, 2.2, 4.8, 2.8]
X2 = [1.0, 3.0, 2.0, 5.0, 4.0, 2.5, 3.5, 1.5, 4.5, 3.2, 2.2, 1.1, 2.0, 3.9]
Y = [5.1, 6.3, 4.8, 7.2, 5.9, 6.6, 5.4, 6.0, 5.5, 6.8, 4.2, 5.0, 12.0, 6.1]


def rel(a, b):
    return abs(a - b) / abs(b)


def test_influence_matches_influence_measures():
    # rstudent, rstandard, dffits, dfbetas, covratio, cooks.distance of
    # lm(y ~ x1 + x2), observations 1 and 13 (the outlier)
    X = np.array([[1.0, a, b] for a, b in zip(X1, X2)])
    inf = D.compute_influence(np.array(Y), X)
    beta = np.linalg.lstsq(X, np.array(Y), rcond=None)[0]
    res = D.compute_residuals(np.array(Y), X @ beta, X)
    pairs = [
        (res.studentized_residuals, [0.914783071043695, 6.0596114804277]),
        (res.standardized_residuals, [0.921644360506835, 2.94032176312096]),
        (inf.dffits, [0.543329498454059, 11.8358008674714]),
        (inf.covratio, [1.41480024713985, 0.0628500511816282]),
        (inf.cooks_distance, [0.099883974741706, 10.9944675572496]),
    ]
    for got, ref in pairs:
        assert rel(float(got[0]), ref[0]) <= 1e-12 and rel(float(got[12]), ref[1]) <= 1e-12
    for a, b in zip([float(v) for v in inf.dfbetas[12]], [-6.87907027545427, 11.0349933350104, -4.33277546179186]):
        assert rel(a, b) <= 1e-12


def test_vif_reset_bp_dw_match_car_and_lmtest():
    X = np.array([[1.0, a, b] for a, b in zip(X1, X2)])
    assert rel(D.compute_vif(np.array([[a, b] for a, b in zip(X1, X2)]))["X0"], 1.03351595716963) <= 1e-12
    r = D.ramsey_reset_test(np.array(Y), X)
    assert rel(r.statistic, 18.8448994488869) <= 1e-12 and rel(r.p_value, 0.000606166983174552) <= 1e-10
    beta = np.linalg.lstsq(X, np.array(Y), rcond=None)[0]
    res = D.compute_residuals(np.array(Y), X @ beta, X)
    assert rel(res.heteroskedasticity_test["statistic"], 6.07553444562654) <= 1e-12
    assert rel(res.autocorrelation_test["durbin_watson"], 3.38472009480018) <= 1e-12
