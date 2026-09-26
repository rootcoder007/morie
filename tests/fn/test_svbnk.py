"""svbnk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svbnk import banks_set


def test_svbnk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        banks_set(data=None)
