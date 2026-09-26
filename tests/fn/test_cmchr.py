"""cmchr is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cmchr import cmchr


def test_cmchr_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cmchr()
