"""kgorv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgorv import ok_variance


def test_kgorv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ok_variance(data=None)
