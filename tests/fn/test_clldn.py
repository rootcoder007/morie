"""clldn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clldn import clldn


def test_clldn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clldn()
