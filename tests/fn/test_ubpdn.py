"""ubpdn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubpdn import ubpdn


def test_ubpdn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubpdn()
