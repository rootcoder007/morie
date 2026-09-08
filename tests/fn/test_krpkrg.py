"""krpkrg: ordinary kriging.

The generated test imported `poisson_kriging`, which this module does not
provide. Rewritten against ordinary_kriging and anchored on exact
interpolation, the defining property of a kriging predictor with no nugget.
"""

from morie.fn import _array_core as np
import pytest

from morie.fn.krpkrg import ordinary_kriging

COORDS = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]
VALUES = [1.0, 2.0, 3.0, 4.0, 2.5]


def test_kriging_interpolates_exactly_at_the_data_locations():
    """With nugget = 0 the predictor honours the data: predicting at an
    observed location returns that observation, with ~zero variance."""
    for i, c in enumerate(COORDS):
        r = ordinary_kriging(COORDS, VALUES, [c], nugget=0.0)
        assert float(np.asarray(r["prediction"])[0]) == pytest.approx(VALUES[i], abs=1e-6)
        assert float(np.asarray(r["variance"])[0]) == pytest.approx(0.0, abs=1e-6)


def test_weights_sum_to_one():
    """Ordinary kriging is unbiased by construction: its weights sum to 1."""
    r = ordinary_kriging(COORDS, VALUES, [[0.25, 0.75]], nugget=0.0)
    w = np.asarray(r["weights"])
    assert float(sum(np.asarray(w)[0] if hasattr(w[0], "__len__") else w)) == pytest.approx(1.0)


def test_prediction_stays_inside_the_data_range():
    r = ordinary_kriging(COORDS, VALUES, [[0.4, 0.6], [0.9, 0.1]], nugget=0.0)
    for p in np.asarray(r["prediction"]):
        assert min(VALUES) - 1e-6 <= float(p) <= max(VALUES) + 1e-6
