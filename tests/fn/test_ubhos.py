"""ubhos is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubhos import ubhos


def test_ubhos_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubhos()
