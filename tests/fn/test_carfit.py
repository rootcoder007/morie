"""carfit is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.carfit import carfit


def test_carfit_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        carfit(ll=None, p_d=None)
