"""baytsm: first-order polynomial dynamic linear model.

The generated test imported `bayes_time_series`, a name that does not
exist. Rewritten against dlm_local_level and anchored on the filter's
defining behaviour rather than on the presence of a dictionary key.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.baytsm import dlm_local_level

Y = [1.0, 1.2, 0.9, 1.1, 1.3, 1.0, 1.15, 0.95]


def test_output_series_match_the_input_length():
    r = dlm_local_level(Y)
    assert len(np.asarray(r["filtered"])) == len(Y)
    assert len(np.asarray(r["smoothed"])) == len(Y)


def test_a_precise_observation_is_followed_closely():
    """Tiny observation variance means the filter trusts the data."""
    r = dlm_local_level(Y, V=1e-6, W=1.0)
    assert float(np.asarray(r["filtered"])[-1]) == pytest.approx(Y[-1], abs=1e-2)


def test_a_noisy_observation_is_smoothed_towards_the_level():
    """Large V, small W is the near-constant-level regime: the filtered
    series must vary less than the data it is filtering."""
    r = dlm_local_level(Y, V=100.0, W=1e-6)
    f = [float(v) for v in np.asarray(r["filtered"])]
    assert max(f) - min(f) < max(Y) - min(Y)


def test_smoothing_is_never_less_certain_than_filtering():
    """The smoother conditions on the whole series, so its variance cannot
    exceed the filter's at the same time point."""
    r = dlm_local_level(Y)
    fv = [float(v) for v in np.asarray(r["filtered_var"])]
    sv = [float(v) for v in np.asarray(r["smoothed_var"])]
    assert all(s <= f + 1e-9 for s, f in zip(sv, fv))
