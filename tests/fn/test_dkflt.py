"""dkflt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkflt import dkflt


def test_dkflt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkflt()
