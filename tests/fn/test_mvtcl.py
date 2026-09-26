"""mvtcl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.mvtcl import mvtcl


def test_mvtcl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        mvtcl()
