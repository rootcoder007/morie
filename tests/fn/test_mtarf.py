"""mtarf is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mtarf import mtarf


def test_mtarf_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mtarf()
