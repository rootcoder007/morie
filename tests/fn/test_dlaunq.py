"""dlaunq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dlaunq import dlaunq


def test_dlaunq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dlaunq()
