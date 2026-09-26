"""dkrgr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkrgr import dkrgr


def test_dkrgr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkrgr()
