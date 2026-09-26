"""sofcl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sofcl import sofcl


def test_sofcl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sofcl()
