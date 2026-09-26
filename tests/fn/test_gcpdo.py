"""gcpdo is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.gcpdo import gcpdo


def test_gcpdo_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gcpdo()
