"""ubgas is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubgas import ubgas


def test_ubgas_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubgas()
