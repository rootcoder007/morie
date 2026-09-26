"""nblnq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.nblnq import nblnq


def test_nblnq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        nblnq()
