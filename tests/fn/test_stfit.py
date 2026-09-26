"""stfit is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stfit import stfit


def test_stfit_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stfit()
