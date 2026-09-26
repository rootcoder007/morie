"""xrcar is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrcar import car_ml


def test_xrcar_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        car_ml(data=None)
