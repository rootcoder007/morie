"""cmvet is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cmvet import cmvet


def test_cmvet_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cmvet()
