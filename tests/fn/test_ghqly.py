"""ghqly is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghqly import ghqly


def test_ghqly_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghqly()
