"""csdrg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.csdrg import csdrg


def test_csdrg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        csdrg()
