"""mddim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mddim import mddim


def test_mddim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mddim()
