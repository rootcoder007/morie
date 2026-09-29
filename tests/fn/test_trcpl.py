"""Tests for morie.fn.trcpl -- trace plot data."""

from morie.fn import _array_core as np
from morie.fn.trcpl import trace_plot_data, trcpl


def test_alias():
    assert trcpl is trace_plot_data


def test_smoke():
    chain = np.random.default_rng(42).standard_normal((100, 3))
    r = trace_plot_data(chain)
    assert r.name == "trace_plot_data"
    assert r.extra["n_params"] == 3
    assert r.extra["n_samples"] == 100
    assert "param_0" in r.extra["traces"]


def test_traces_are_the_chain_columns():
    chain = [[0.1, 5.0], [0.3, 4.0], [0.2, 4.5]]
    r = trace_plot_data(chain)
    assert r.extra["traces"] == {"param_0": [0.1, 0.3, 0.2], "param_1": [5.0, 4.0, 4.5]}
    assert r.extra["iterations"] == [0, 1, 2] and r.value == 3.0
