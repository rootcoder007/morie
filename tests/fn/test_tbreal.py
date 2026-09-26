"""tbreal is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tbreal import tbreal


def test_tbreal_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tbreal()
