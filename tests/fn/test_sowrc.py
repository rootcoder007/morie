"""sowrc is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.sowrc import sowrc


def test_sowrc_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        sowrc()
