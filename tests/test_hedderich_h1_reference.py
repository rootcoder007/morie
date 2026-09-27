"""Hedderich methods (batch 1) against R references: DescTools, qt, cor.test, psych, pwr, rmcorr."""

from morie.fn.corcmp import compare_correlations
from morie.fn.corrho import correlation_test
from morie.fn.corss import correlation_sample_size
from morie.fn.corwil import williams_test
from morie.fn.gktau import goodman_kruskal_tau
from morie.fn.normtl import normal_tolerance_factor
from morie.fn.rpmcor import repeated_measures_correlation


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_goodman_kruskal_tau_matches_desctools_and_book():
    r = goodman_kruskal_tau([[10, 30, 5], [0, 20, 30], [5, 0, 0]])
    # DescTools::GoodmanKruskalTau(direction = "column" / "row"); book p. 78: 0.236 and 0.28
    assert close(r["tau_col_given_row"], 0.23599632690541772)
    assert close(r["tau_row_given_col"], 0.28003494975972054)


def test_tolerance_factor_matches_qt_and_table_6_21():
    # qt(confidence, n - 1, qnorm(coverage) sqrt(n)) / sqrt(n); Howe's two-sided formula
    assert close(normal_tolerance_factor(10, 0.95, 0.95, "one")["k"], 2.9109634130810349, 1e-10)
    assert close(normal_tolerance_factor(30, 0.90, 0.99, "one")["k"], 2.0298341901276977, 1e-10)
    assert close(normal_tolerance_factor(10, 0.95, 0.95, "two")["k"], 3.3819134905090209)
    assert close(normal_tolerance_factor(30, 0.90, 0.99, "two")["k"], 2.3847381969510941)


def test_correlation_tests():
    r = correlation_test(0.84011830527061726, 10)
    # cor.test on the book's x, y (p. 767)
    assert close(r["statistic"], 4.3808985556112692) and close(r["p_value"], 0.0023459622144385321, 1e-10)
    # (atanh r - atanh rho0) sqrt(n - 3); book examples 3.085, 0.694, 3.296
    assert close(correlation_test(0.966, 14, 0.8)["statistic"], 3.0847449560475293)
    assert close(correlation_test(0.3, 40, 0.4)["statistic"], -0.69422158774314413)
    assert close(correlation_test(0.97, 14, 0.8)["statistic"], 3.2956751022366437)
    assert close(correlation_test(0.966, 14, 0.8, method="samiuddin")["statistic"], 3.7069458788857004)


def test_compare_correlations():
    r = compare_correlations([0.6, 0.7, 0.8], [28, 33, 23])
    # book p. 779: chi2 = 1.83, r = 0.702, 0.57 <= rho <= 0.80
    assert close(r["chi2"], 1.8273479552539205) and close(r["r_pooled"], 0.70184762492712993)
    assert close(r["ci"][0], 0.56803393224890819) and close(r["ci"][1], 0.799508931302501)
    r = compare_correlations([0.6, 0.8], [28, 23])
    # psych::r.test(n = 28, r12 = 0.6, r34 = 0.8, n2 = 23)
    assert close(r["z_two"], 1.3515503603605483) and close(r["p_two"], 0.17651919922959225, 1e-10)
    assert close(r["ci"][0], 0.5235223424364438) and close(r["ci"][1], 0.82283328714253823)
    r = compare_correlations([0.422, 0.388, 0.569], [30, 30, 30])
    assert close(r["r_gem"], 0.45966666666666667) and close(r["t_gem"], 4.7999256665164065)


def test_williams_matches_psych_r_test():
    r = williams_test(0.85, 0.71, 0.80, 30)
    # psych::r.test(n = 30, r12 = 0.85, r13 = 0.71, r23 = 0.80)
    assert close(r["statistic"], 2.168701837454599) and close(r["p_value"], 0.039082529371308745, 1e-10)


def test_correlation_sample_size():
    r = correlation_sample_size(0.6, power=0.9)
    assert r["n"] == 25 and close(r["n_exact"], 24.869824430385489)
    assert correlation_sample_size(0.2, power=0.9)["n"] == 259
    assert close(correlation_sample_size(0.6, n=25)["power"], 0.90168014141326802)


def test_repeated_measures_correlation_matches_rmcorr():
    x = [
        47,
        46,
        50,
        52,
        46,
        36,
        47,
        46,
        36,
        44,
        49,
        50,
        42,
        48,
        60,
        47,
        51,
        57,
        49,
        49,
        51,
        46,
        46,
        45,
        52,
        54,
        48,
        47,
        47,
        54,
        63,
        70,
        63,
        58,
        59,
        61,
        67,
        64,
        59,
        61,
    ]
    y = [
        51,
        53,
        57,
        54,
        55,
        53,
        54,
        57,
        61,
        57,
        52,
        56,
        46,
        52,
        53,
        49,
        52,
        50,
        50,
        49,
        46,
        48,
        47,
        55,
        49,
        61,
        53,
        48,
        50,
        44,
        64,
        62,
        66,
        64,
        62,
        62,
        58,
        62,
        67,
        59,
    ]
    s = [i // 10 for i in range(40)]
    r = repeated_measures_correlation(x, y, s)
    # book p. 809: between 0.7720022; rmcorr::rmcorr: r = -0.0211820, p = 0.901, df = 35
    assert close(r["r_between"], 0.77200223038047033)
    assert close(r["r_within"], -0.021182045737978591, 1e-10)
    assert close(r["p_within"], 0.90096938110056957, 1e-10) and r["df_within"] == 35
