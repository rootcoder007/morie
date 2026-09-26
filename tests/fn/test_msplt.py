"""msplt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.msplt import mds_polarity


def test_msplt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mds_polarity(X=None)
