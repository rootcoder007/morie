"""tmbrk is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tmbrk import tmbrk


def test_tmbrk_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tmbrk()
