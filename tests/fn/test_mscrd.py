"""mscrd is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mscrd import mds_coords


def test_mscrd_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mds_coords(X=None)
