import math

from morie.fn.probresults import (
    ar_spectral_density,
    berkson_correlation,
    beta_binomial_pmf,
    bivariate_normal_dependence,
    erlang_renewal_function,
    ma_autocorrelation,
    maximal_correlation,
    mills_conditional_mean,
    simple_epidemic_duration,
    spectral_density_from_acf,
    stirling_gamma_ratio,
    telegraph_process,
    wiener_conditional_correlation,
)


def _simpson(fn, a, b, n):
    h = (b - a) / n
    s = fn(a) + fn(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * fn(a + i * h)
    return s * h / 3


def test_erlang_renewal():
    lam, t = 1.7, 0.9
    assert abs(erlang_renewal_function(t, lam) - (lam * t / 2 - (1 - math.exp(-2 * lam * t)) / 4)) < 1e-13
    assert abs(erlang_renewal_function(t, lam, 1) - lam * t) < 1e-13
    for k in (3, 4, 6):  # transients vanish: m(t) = lam t / k - (k - 1) / (2k)
        assert abs(erlang_renewal_function(80.0, 1.0, k) - (80.0 / k - (k - 1) / (2 * k))) < 1e-10


def test_berkson_by_enumeration():
    g, a, c = 0.13, 0.21, 0.47
    px = py = pxy = 0.0
    for C in (0, 1):
        for H1 in (0, 1):
            for H2 in (0, 1):
                pr = (g if C else 1 - g) * ((c if H1 else 1 - c) if C else (1.0 if H1 == 0 else 0.0)) * (a if H2 else 1 - a)
                X = 1 if (H1 or H2) else 0
                Y = 1 if (C and X) else 0
                px += pr * X
                py += pr * Y
                pxy += pr * X * Y
    rho = (pxy - px * py) / math.sqrt(px * (1 - px) * py * (1 - py))
    assert abs(berkson_correlation(g, a, c) - rho) < 1e-12


def test_mills_by_quadrature():
    x, rho = 0.4, -0.35
    num = _simpson(lambda u: u * math.exp(-0.5 * u * u) / math.sqrt(2 * math.pi), x, x + 20.0, 20000)
    den = 0.5 * math.erfc(x / math.sqrt(2))
    assert abs(mills_conditional_mean(x, rho) - rho * num / den) < 1e-9


def test_bivariate_normal_dependence_quadrature():
    rho = 0.45
    r = bivariate_normal_dependence(rho)
    h, acc = 0.05, 0.0
    c = 1 / (2 * math.pi * math.sqrt(1 - rho * rho))
    for i in range(400):
        x = -10 + (i + 0.5) * h
        for j in range(400):
            y = -10 + (j + 0.5) * h
            f = c * math.exp(-(x * x - 2 * rho * x * y + y * y) / (2 * (1 - rho * rho)))
            gh = math.exp(-0.5 * (x * x + y * y)) / (2 * math.pi)
            acc += f * f / gh * h * h
    assert abs(r.phi2 - (acc - 1)) < 1e-9
    assert abs(r.mutual_information + 0.5 * math.log(1 - rho * rho)) < 1e-15


def test_maximal_correlation():
    t = [[0.3, 0.2], [0.1, 0.4]]
    phi = (0.3 * 0.4 - 0.2 * 0.1) / math.sqrt(0.5 * 0.5 * 0.4 * 0.6)
    assert abs(maximal_correlation(t).m - abs(phi)) < 1e-12
    T = [[5, 1, 2, 7], [1, 6, 3, 2], [4, 2, 8, 1]]
    r = maximal_correlation(T)
    n = sum(sum(row) for row in T)
    pr = [sum(row) / n for row in T]
    pc = [sum(T[i][j] for i in range(3)) / n for j in range(4)]
    ef = sum(pr[i] * r.f[i] for i in range(3))
    vf = sum(pr[i] * r.f[i] ** 2 for i in range(3))
    vg = sum(pc[j] * r.g[j] ** 2 for j in range(4))
    cov = sum(T[i][j] / n * r.f[i] * r.g[j] for i in range(3) for j in range(4))
    assert abs(ef) < 1e-12 and abs(vf - 1) < 1e-12 and abs(vg - 1) < 1e-12
    assert abs(cov - r.m) < 1e-12
    ind = [[pr[i] * pc[j] for j in range(4)] for i in range(3)]
    assert maximal_correlation(ind).m < 1e-6


def test_beta_binomial():
    n, a, b = 9, 1.7, 2.6
    tot = 0.0
    for k in range(n + 1):
        direct = math.comb(n, k) * math.gamma(a + k) * math.gamma(n + b - k) * math.gamma(a + b)
        direct /= math.gamma(a) * math.gamma(b) * math.gamma(n + a + b)
        assert abs(beta_binomial_pmf(k, n, a, b) - direct) < 1e-12
        tot += beta_binomial_pmf(k, n, a, b)
    assert abs(tot - 1) < 1e-12


def test_ar_and_acf_spectra():
    al, lam = 0.6, 1.3
    ar1 = (1 - al * al) / (2 * math.pi * (1 - 2 * al * math.cos(lam) + al * al))
    assert abs(ar_spectral_density(lam, [al]) - ar1) < 1e-12
    assert abs(spectral_density_from_acf(lam, [al**n for n in range(1, 200)]) - ar1) < 1e-12
    p1, p2, s2 = 0.5, -0.3, 2.0  # AR(2) variance from Yule-Walker
    g0 = s2 * (1 - p2) / ((1 + p2) * ((1 - p2) ** 2 - p1 * p1))
    un = ar_spectral_density(lam, [p1, p2], s2, normalized=False)
    assert abs(ar_spectral_density(lam, [p1, p2], s2) - un / g0) < 1e-12


def test_telegraph_fourier():
    a, b, lam = 0.7, 1.1, 0.9
    r = telegraph_process(0.3, lam, a, b)
    assert abs(r.rho - math.exp(-0.3 * (a + b))) < 1e-15
    # f(lam) = (1/pi) int_0^inf rho(t) cos(lam t) dt
    acc = _simpson(lambda t: math.exp(-t * (a + b)) * math.cos(lam * t), 0.0, 40.0, 40000)
    assert abs(r.f - acc / math.pi) < 1e-9


def test_wiener_schur():
    s, t, u, v = 0.5, 1.2, 2.0, 3.1
    T = [t, u, s, v]
    C = [[min(p, q) for q in T] for p in T]
    d = C[2][2] * C[3][3] - C[2][3] ** 2
    inv = [[C[3][3] / d, -C[2][3] / d], [-C[2][3] / d, C[2][2] / d]]
    cond = [[C[i][j] - sum(C[i][2 + k] * inv[k][m] * C[2 + m][j] for k in range(2) for m in range(2)) for j in range(2)] for i in range(2)]
    assert abs(wiener_conditional_correlation(s, t, u, v) - cond[0][1] / math.sqrt(cond[0][0] * cond[1][1])) < 1e-12


def test_epidemic_stirling_ma():
    N, lam = 7, 0.3
    r = simple_epidemic_duration(N, lam)
    assert abs(r.mean - sum(1 / (lam * i * (N + 1 - i)) for i in range(1, N + 1))) < 1e-12
    assert abs(r.variance - sum(1 / (lam * i * (N + 1 - i)) ** 2 for i in range(1, N + 1))) < 1e-12
    for t in (2.5, 10.0, 40.0):
        assert abs(stirling_gamma_ratio(t) - math.gamma(t) / (t ** (t - 0.5) * math.exp(-t) * math.sqrt(2 * math.pi))) < 1e-12
    a = -0.7
    assert abs(ma_autocorrelation([a], 3)[1] - a / (1 + a * a)) < 1e-15
