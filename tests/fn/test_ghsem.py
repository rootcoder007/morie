"""ghsem is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghsem import ghsem


def test_ghsem_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghsem()
