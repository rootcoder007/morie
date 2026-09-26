"""msdim is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msdim import mds_dims


def test_msdim_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mds_dims(X=None)
