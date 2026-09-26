"""kgckv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgckv import cok_variance


def test_kgckv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cok_variance(data=None)
