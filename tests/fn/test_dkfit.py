"""dkfit is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkfit import dkfit


def test_dkfit_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkfit()
