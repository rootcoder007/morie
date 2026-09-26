"""swmtrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.swmtrc import swmtrc


def test_swmtrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        swmtrc(W=None)
