"""endep is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.endep import endep


def test_endep_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        endep()
