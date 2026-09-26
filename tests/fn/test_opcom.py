"""opcom is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opcom import opcom


def test_opcom_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opcom()
