"""opvhc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opvhc import opvhc


def test_opvhc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opvhc()
