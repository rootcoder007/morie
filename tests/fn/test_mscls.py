"""mscls is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mscls import classical_mds


def test_mscls_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        classical_mds(X=None)
