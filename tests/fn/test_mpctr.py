"""mpctr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mpctr import mpctr


def test_mpctr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mpctr()
