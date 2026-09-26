"""ubpkn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubpkn import ubpkn


def test_ubpkn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubpkn()
