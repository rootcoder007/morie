"""opnwt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opnwt import opnwt


def test_opnwt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opnwt()
