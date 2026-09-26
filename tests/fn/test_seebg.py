"""seebg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.seebg import seebg


def test_seebg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        seebg()
