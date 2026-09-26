"""ghbym is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghbym import ghbym


def test_ghbym_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghbym()
