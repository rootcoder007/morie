"""msscr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msscr import mds_scree


def test_msscr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mds_scree(X=None)
