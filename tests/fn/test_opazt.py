"""opazt is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.opazt import opazt


def test_opazt_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        opazt()
