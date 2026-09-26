"""msrsq is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msrsq import mds_rsq


def test_msrsq_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mds_rsq(X=None)
