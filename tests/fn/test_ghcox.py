"""ghcox is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghcox import ghcox


def test_ghcox_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghcox()
