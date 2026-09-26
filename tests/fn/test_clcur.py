"""clcur is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.clcur import clcur


def test_clcur_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        clcur()
