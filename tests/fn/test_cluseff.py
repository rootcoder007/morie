"""Tests for cluseff.intracluster_correlation_rho."""

from morie.fn import _array_core as np

from morie.fn.cluseff import intracluster_correlation_rho


def test_cluseff_basic():
    """Test basic functionality."""
    rng_y = np.random.default_rng(43)
    rng_c = np.random.default_rng(42)
    a = 5
    n_per = 20
    n = a * n_per
    # Draw cluster-level random effects and within-cluster noise so that
    # there is genuine between-cluster variation.
    effects = rng_y.normal(0.0, 1.0, a)
    noise = rng_c.normal(0.0, 0.5, n)
    y = np.concatenate(
        [effects[i] + noise[i * n_per:(i + 1) * n_per] for i in range(a)]
    )
    cluster = np.concatenate(
        [np.full(n_per, i, dtype=int) for i in range(a)]
    )
    result = intracluster_correlation_rho(y, cluster)

    # Independent recomputation of every returned statistic from the
    # documented formula, so the assertions are not a copy of the
    # function's output.
    v = [float(t) for t in np.asarray(y, dtype=float).ravel().tolist()]
    g = [int(t) for t in np.asarray(cluster).ravel().tolist()]
    groups = {}
    for val, lab in zip(v, g):
        groups.setdefault(lab, []).append(val)
    a_chk = len(groups)
    n_tot = len(v)
    grand = sum(v) / n_tot
    sizes = [len(vals) for vals in groups.values()]
    ssb = sum(
        len(vals) * (sum(vals) / len(vals) - grand) ** 2
        for vals in groups.values()
    )
    ssw = sum(
        sum((t - sum(vals) / len(vals)) ** 2 for t in vals)
        for vals in groups.values()
    )
    msb = ssb / (a_chk - 1)
    msw = ssw / (n_tot - a_chk)
    n0 = (n_tot - sum(s * s for s in sizes) / n_tot) / (a_chk - 1)
    var_a = (msb - msw) / n0
    rho = var_a / (var_a + msw)
    nbar = n_tot / a_chk
    deff = 1.0 + (nbar - 1.0) * rho

    # The RichResult is dict-like.
    assert isinstance(result, dict)

    # Required output keys from the docstring.
    for key in (
        "rho",
        "deff",
        "msb",
        "msw",
        "n0",
        "var_between",
        "n_clusters",
        "n_obs",
        "mean_cluster_size",
        "effective_n",
    ):
        assert key in result, f"missing key: {key}"

    # Shape / bookkeeping values.
    assert result["n_clusters"] == a_chk
    assert result["n_obs"] == n_tot
    assert result["mean_cluster_size"] == float(nbar)

    # Formula-based numeric expectations.
    assert abs(result["msb"] - msb) < 1e-12
    assert abs(result["msw"] - msw) < 1e-12
    assert abs(result["n0"] - n0) < 1e-12
    assert abs(result["var_between"] - var_a) < 1e-12
    assert abs(result["rho"] - rho) < 1e-12
    assert abs(result["deff"] - deff) < 1e-12
    assert abs(result["effective_n"] - n_tot / deff) < 1e-12


def test_cluseff_edge():
    """Test edge cases."""
    rng_y = np.random.default_rng(43)
    rng_c = np.random.default_rng(42)
    a = 5
    n_per = 20
    n = a * n_per
    effects = rng_y.normal(0.0, 1.0, a)
    noise = rng_c.normal(0.0, 0.5, n)
    y = np.concatenate(
        [effects[i] + noise[i * n_per:(i + 1) * n_per] for i in range(a)]
    )
    cluster = np.concatenate(
        [np.full(n_per, i, dtype=int) for i in range(a)]
    )
    result = intracluster_correlation_rho(y, cluster)
    assert isinstance(result, dict)
