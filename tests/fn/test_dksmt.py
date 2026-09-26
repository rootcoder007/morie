"""dksmt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dksmt import dksmt


def test_dksmt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dksmt()
