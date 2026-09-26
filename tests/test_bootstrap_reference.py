"""morie.bootstrap_methods interval rules against boot::boot.ci."""

import math

from morie import bootstrap_methods as B
from morie.fn import _stats_core as stats

X = [5.1, 6.3, 4.8, 7.2, 5.9, 6.6, 5.4, 6.0, 5.5, 6.8, 4.2, 5.0, 3.9, 5.8, 4.4]


def _lcg_index(n, R):
    # the same pseudo-random index matrix built in R for the reference values
    st, out = 12345, []
    for _ in range(R):
        row = []
        for _ in range(n):
            st = (69069 * st + 1) % 4294967296
            row.append((st // 65536) % n)
        out.append(row)
    return out


def test_norm_inter_and_regression_influence_match_boot():
    # boot:::norm.inter and a regression of the replicates on the resampling
    # proportions (boot::empinf type "reg") on this index matrix
    idx = _lcg_index(len(X), 199)
    f = lambda d: sum(v * v for v in d) / len(d)  # noqa: E731
    boot = [f([X[i] for i in ix]) for ix in idx]
    t0 = f(X)
    q = B._norm_inter(boot, [0.05, 0.95])
    assert abs(q[0] - 26.277333333333335) <= 1e-12 and abs(q[1] - 35.368000000000002) <= 1e-12
    L = B._empinf_reg(idx, boot, len(X))
    acc = sum(v**3 for v in L) / (6 * sum(v * v for v in L) ** 1.5)
    assert abs(acc - 0.011511851008016017) <= 1e-12
    w = stats.norm.ppf(sum(1 for v in boot if v < t0) / len(boot))
    adj = [stats.norm.cdf(w + (w + z) / (1 - acc * (w + z))) for z in (stats.norm.ppf(0.05), stats.norm.ppf(0.95))]
    bca = B._norm_inter(boot, adj)
    assert abs(bca[0] - 25.934217498719672) <= 1e-10 and abs(bca[1] - 35.202055783084219) <= 1e-10


def test_bootstrap_reports_acceleration_and_studentizes_per_replicate():
    r = B.bootstrap(X, lambda d: sum(v * v for v in d) / len(d), n_boot=499, ci_method="bca", seed=1)
    m2 = sum(v * v for v in X) / len(X)
    L = [v * v - m2 for v in X]
    assert abs(r.acceleration - sum(v**3 for v in L) / (6 * sum(v * v for v in L) ** 1.5)) <= 1e-10
    s = B.bootstrap(X, lambda d: sum(d) / len(d), n_boot=199, ci_method="studentized", seed=3)
    assert s.ci_lower < sum(X) / len(X) < s.ci_upper
    cl = [i // 3 for i in range(15)]
    cb = B.bootstrap(X, lambda d: sum(d) / len(d), n_boot=299, ci_method="bca", seed=5, cluster=cl)
    jack = [sum(v for v, c in zip(X, cl) if c != g) / 12 for g in range(5)]
    jm = sum(jack) / 5
    Lc = [4 * (jm - v) for v in jack]
    assert abs(cb.acceleration - sum(v**3 for v in Lc) / (6 * sum(v * v for v in Lc) ** 1.5)) <= 1e-12
    assert math.isfinite(cb.ci_lower) and cb.ci_lower < cb.ci_upper
