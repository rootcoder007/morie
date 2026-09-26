"""ghexp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghexp import ghexp


def test_ghexp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghexp()
