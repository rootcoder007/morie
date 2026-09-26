"""ubgrn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubgrn import ubgrn


def test_ubgrn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubgrn()
