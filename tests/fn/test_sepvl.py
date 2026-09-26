"""sepvl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sepvl import sepvl


def test_sepvl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sepvl()
