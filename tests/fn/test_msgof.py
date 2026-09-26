"""msgof is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msgof import mds_gof


def test_msgof_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mds_gof(X=None)
