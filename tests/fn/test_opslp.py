"""opslp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opslp import opslp


def test_opslp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opslp()
