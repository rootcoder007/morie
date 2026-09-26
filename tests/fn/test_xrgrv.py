"""xrgrv is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.xrgrv import gravity_model


def test_xrgrv_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        gravity_model(data=None)
