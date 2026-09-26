"""svsut is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.svsut import svsut


def test_svsut_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        svsut()
