"""afsed is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.afsed import afsed


def test_afsed_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        afsed()
