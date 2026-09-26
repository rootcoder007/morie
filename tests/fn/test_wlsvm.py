"""wlsvm is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.wlsvm import wlsvm


def test_wlsvm_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        wlsvm()
