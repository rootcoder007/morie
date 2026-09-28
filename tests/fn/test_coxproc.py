import math

from morie.fn.coxproc import lgcp_moments, lgcp_simulate, thomas_pcf, thomas_simulate, voronoi_residuals


def _z(counts, mean):
    n = len(counts)
    m = sum(counts) / n
    sd = math.sqrt(sum((c - m) ** 2 for c in counts) / (n - 1))
    return (m - mean) / (sd / math.sqrt(n))


def test_lgcp_mean_count():
    model = {"model": "Exp", "psill": 0.4, "range": 0.3}
    counts = [len(lgcp_simulate((0, 1, 0, 1), 3, 2.5, model, seed=s).points) for s in range(300)]
    assert abs(_z(counts, lgcp_moments(2.5, 0.4, 1.0).expected_count)) < 4
    r = lgcp_moments(1.0, 0.4, 1.0, 0.2, model)
    assert abs(r.pcf - math.exp(0.4 * math.exp(-0.2 / 0.3))) < 1e-15


def test_thomas_mean_count_and_k():
    counts = [len(thomas_simulate(10, 0.05, 5, (0, 1, 0, 1), seed=s).points) for s in range(300)]
    assert abs(_z(counts, 50.0)) < 4
    kappa, s, R = 10.0, 0.05, 0.12
    n = 2000
    h = R / n
    acc = 0.0
    for i in range(n + 1):  # Simpson: K(R) = int_0^R 2 pi r g(r) dr
        r = i * h
        f = 2 * math.pi * r * thomas_pcf(r, kappa, s).pcf
        acc += f * (1 if i in (0, n) else (4 if i % 2 else 2))
    assert abs(acc * h / 3 - thomas_pcf(R, kappa, s).K) < 1e-10


def test_voronoi_residuals_integrate_the_intensity():
    pts = [(0.1 + 0.8 * ((i * 0.618034) % 1), 0.1 + 0.8 * ((i * 0.381966 + 0.2) % 1)) for i in range(15)]
    r = voronoi_residuals(pts, (0, 0, 1, 1), [math.log(15.0), 0.0, 0.0])
    assert all(abs(res - (1 - 15.0 * a)) < 1e-12 for res, a in zip(r.residuals, r.areas))
    b = [0.5, 0.8, -0.6]
    r = voronoi_residuals(pts, (0, 0, 1, 1), b)
    total = math.exp(b[0]) * (math.exp(b[1]) - 1) / b[1] * (math.exp(b[2]) - 1) / b[2]
    assert abs(sum(1 - v for v in r.residuals) - total) < 1e-7
