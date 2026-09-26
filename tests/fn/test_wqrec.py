"""wqrec is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wqrec import wqrec


def test_wqrec_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wqrec()
