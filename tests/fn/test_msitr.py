"""msitr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msitr import mds_iter


def test_msitr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mds_iter(X=None)
