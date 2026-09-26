"""fofrag is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.fofrag import fofrag


def test_fofrag_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        fofrag()
