"""chlkrg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.chlkrg import chlkrg


def test_chlkrg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        chlkrg()
