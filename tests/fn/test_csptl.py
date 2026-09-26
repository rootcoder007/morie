"""csptl is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csptl import csptl


def test_csptl_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csptl()
