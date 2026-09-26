"""isvlt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.isvlt import isvlt


def test_isvlt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        isvlt()
