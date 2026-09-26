"""msjck is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msjck import mds_jackknife


def test_msjck_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mds_jackknife(X=None)
