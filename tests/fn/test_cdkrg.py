"""cdkrg is a placeholder: it must refuse to run, not return a number."""

import pytest

from morie.fn.cdkrg import cdkrg


def test_cdkrg_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="not implemented"):
        cdkrg()
