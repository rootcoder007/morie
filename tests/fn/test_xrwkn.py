"""xrwkn is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrwkn import w_knn


def test_xrwkn_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        w_knn(data=None)
