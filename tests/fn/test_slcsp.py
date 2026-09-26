"""slcsp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.slcsp import slcsp


def test_slcsp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        slcsp()
