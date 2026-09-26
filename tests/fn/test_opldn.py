"""opldn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opldn import opldn


def test_opldn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opldn()
