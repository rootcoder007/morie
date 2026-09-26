"""tmspln is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.tmspln import tmspln


def test_tmspln_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        tmspln()
