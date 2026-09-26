"""ghgwr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghgwr import ghgwr


def test_ghgwr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghgwr()
