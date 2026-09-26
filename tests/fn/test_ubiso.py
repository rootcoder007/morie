"""ubiso is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubiso import ubiso


def test_ubiso_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubiso()
