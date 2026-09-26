"""ublit is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ublit import ublit


def test_ublit_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ublit()
