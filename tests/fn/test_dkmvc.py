"""dkmvc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.dkmvc import dkmvc


def test_dkmvc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        dkmvc()
