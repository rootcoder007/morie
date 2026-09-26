"""ubfod is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ubfod import ubfod


def test_ubfod_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ubfod()
