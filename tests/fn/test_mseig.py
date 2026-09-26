"""mseig is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mseig import mds_eigen


def test_mseig_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mds_eigen(X=None)
