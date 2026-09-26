"""msflp is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msflp import mds_flip


def test_msflp_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mds_flip(X=None)
