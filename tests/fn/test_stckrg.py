"""stckrg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.stckrg import stckrg


def test_stckrg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        stckrg()
