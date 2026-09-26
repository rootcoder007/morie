"""ghke2 is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.ghke2 import ghke2


def test_ghke2_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        ghke2()
