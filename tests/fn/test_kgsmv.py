"""kgsmv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgsmv import sk_variance


def test_kgsmv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sk_variance(data=None)
