"""kgorw is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.kgorw import ok_weights


def test_kgorw_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ok_weights(data=None)
