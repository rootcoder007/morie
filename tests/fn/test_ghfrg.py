"""ghfrg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghfrg import ghfrg


def test_ghfrg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghfrg()
