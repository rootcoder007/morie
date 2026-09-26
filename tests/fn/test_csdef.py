"""csdef is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csdef import csdef


def test_csdef_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csdef()
